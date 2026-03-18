"""
Wildcard Tracer — SD WebUI Extension (reForge / AUTOMATIC1111)
Intercepts sd-dynamic-prompts wildcard resolution and writes the full
nested resolution tree to image PNG metadata as "Wildcard chain".

Format: ~~wc~~<<chosen entry with ~~sub~~<<...>> annotations>>
"""

import modules.scripts as scripts
import re
import threading


# ---------------------------------------------------------------------------
# Resolution recorder
# ---------------------------------------------------------------------------

class _ResolutionRecorder:
    def __init__(self):
        self._patched  = False
        self._records  = None   # flat list while active, None when idle
        self._lock     = threading.Lock()

    def start(self):
        with self._lock:
            self._records = []
        self._ensure_patched()

    def record(self, wc_name, raw_value):
        """
        Record a wildcard resolution.
        wc_name   — the wildcard name (e.g. 'ill')
        raw_value — the raw chosen line from the .txt file,
                    with any ~~sub-wildcards~~ still present and unresolved.
        """
        with self._lock:
            if self._records is not None:
                self._records.append((wc_name, raw_value))
                print(f"[WildcardTracer] RECORDED: {wc_name!r} -> {repr(str(raw_value))[:80]}")

    def snapshot(self):
        with self._lock:
            return list(self._records) if self._records is not None else []

    def finish(self):
        with self._lock:
            recs = self._records if self._records is not None else []
            self._records = None
            return recs

    def _ensure_patched(self):
        if self._patched:
            return
        self._patched = True
        self._patch()

    def _patch(self):
        """
        Patch RandomSampler._get_wildcard so we record the raw chosen value
        (the line from the .txt file, sub-wildcards still present) BEFORE
        context.sample_prompts() recurses into it.

        The original body is:
            wildcard_path = next(iter(context.sample_prompts(command.wildcard, 1))).text
            context = context.with_variables(command.variables)
            values  = context.wildcard_manager.get_values(wildcard_path)
            gen     = self._get_wildcard_choice_generator(context, values)
            while True:
                value = next(gen)                        # raw string from .txt
                yield from context.sample_prompts(value, 1)  # recurses into sub-wcs

        We re-implement this body, inserting recorder.record(wildcard_path, raw_value)
        immediately after `raw_value = next(gen)` and before sample_prompts recurses.
        """
        recorder = self

        try:
            from dynamicprompts.samplers.base import Sampler
            from dynamicprompts.samplers.random import RandomSampler
            import inspect
            methods = sorted(
                n for n, _ in inspect.getmembers(RandomSampler, predicate=inspect.isfunction)
                if not n.startswith('__')
            )
            print(f"[WildcardTracer] RandomSampler methods: {methods}")
        except Exception as e:
            print(f"[WildcardTracer] Could not import samplers: {e}")
            return

        # ------------------------------------------------------------------
        # Primary patch on RandomSampler._get_wildcard
        # ------------------------------------------------------------------
        if not hasattr(RandomSampler, '_get_wildcard'):
            print("[WildcardTracer] _get_wildcard not on RandomSampler -- skipping primary patch")
        else:
            orig_get_wildcard = RandomSampler._get_wildcard

            def patched_get_wildcard(self_s, command, context):
                wildcard_path = next(
                    iter(context.sample_prompts(command.wildcard, 1))
                ).text
                ctx2   = context.with_variables(command.variables)
                values = ctx2.wildcard_manager.get_values(wildcard_path)

                if len(values) == 0:
                    # No entries in this wildcard file. Record a placeholder so
                    # _count_records_for accounts for this wildcard slot correctly
                    # when computing per-image record boundaries.
                    try:
                        recorder.record(wildcard_path, "")
                    except Exception:
                        pass
                    yield from orig_get_wildcard(self_s, command, context)
                    return

                gen = self_s._get_wildcard_choice_generator(ctx2, values)
                while True:
                    raw_value = next(gen)   # raw string, sub-wildcards intact
                    try:
                        recorder.record(wildcard_path, raw_value)
                    except Exception as ex:
                        print(f"[WildcardTracer] record error: {ex}")
                    yield from ctx2.sample_prompts(raw_value, 1)

            RandomSampler._get_wildcard = patched_get_wildcard
            print("[WildcardTracer] Patched RandomSampler._get_wildcard")

        # ------------------------------------------------------------------
        # Fallback patch on base Sampler -- covers any non-Random subclass.
        # ------------------------------------------------------------------
        if hasattr(Sampler, '_get_wildcard'):
            orig_base = Sampler._get_wildcard

            def patched_base_get_wildcard(self_s, command, context):
                own = type(self_s).__dict__.get('_get_wildcard')
                if own is not None and own is not patched_base_get_wildcard:
                    yield from own(self_s, command, context)
                    return
                try:
                    wildcard_path = next(
                        iter(context.sample_prompts(command.wildcard, 1))
                    ).text
                    ctx2   = context.with_variables(command.variables)
                    values = ctx2.wildcard_manager.get_values(wildcard_path)
                    if len(values) == 0:
                        try:
                            recorder.record(wildcard_path, "")
                        except Exception:
                            pass
                        yield from orig_base(self_s, command, context)
                        return
                    gen = self_s._get_wildcard_choice_generator(ctx2, values)
                    while True:
                        raw_value = next(gen)
                        try:
                            recorder.record(wildcard_path, raw_value)
                        except Exception as ex:
                            print(f"[WildcardTracer] record error (base): {ex}")
                        yield from ctx2.sample_prompts(raw_value, 1)
                except Exception:
                    yield from orig_base(self_s, command, context)

            Sampler._get_wildcard = patched_base_get_wildcard
            print("[WildcardTracer] Patched base Sampler._get_wildcard (fallback)")

        print("[WildcardTracer] Patch complete.")


RECORDER = _ResolutionRecorder()

# Apply patches at import time -- before any generation hook fires.
try:
    RECORDER._ensure_patched()
except Exception as _patch_ex:
    print(f"[WildcardTracer] Early patch failed (will retry on first generation): {_patch_ex}")


# ---------------------------------------------------------------------------
# Chain builder  (do not modify -- algorithm is correct)
# ---------------------------------------------------------------------------

WC_RE = re.compile(r'~~([^\s]+?)~~')


def build_chain_from_records(template, records):
    """
    Convert flat depth-first records into a nested ~~wc~~<<...>> annotation.

    Example:
      template = '~~ill~~'
      records  = [('ill', '~~directdef~~'),
                  ('directdef', '~~art~~ some text ~~emo~~'),
                  ('art', 'lora stuff'),
                  ('emo', 'uneven eyes')]
    Result:
      '~~ill~~<<~~directdef~~<<~~art~~<<lora stuff>> some text ~~emo~~<<uneven eyes>>>>>>'
    """
    if not records:
        return ""

    it = iter(records)

    def next_rec():
        try:    return next(it)
        except StopIteration: return None

    def annotate(wc_name, value):
        value = str(value)
        parts = WC_RE.split(value)
        inner = ""
        for i, p in enumerate(parts):
            if i % 2 == 0:
                inner += p
            else:
                rec = next_rec()
                if rec is None:
                    inner += f"~~{p}~~"
                else:
                    inner += annotate(rec[0], rec[1])
        return f"~~{wc_name}~~<<{inner}>>"

    parts  = WC_RE.split(template)
    result = ""
    for i, p in enumerate(parts):
        if i % 2 == 0:
            result += p
        else:
            rec = next_rec()
            if rec is None:
                result += f"~~{p}~~"
            else:
                result += annotate(rec[0], rec[1])
    return result


def _count_records_for(template, records, start):
    """
    Walk the record list from `start`, consuming exactly the records that belong
    to one resolution of `template`, and return the number consumed.

    This correctly handles variable depth: each wildcard value can contain a
    different number of embedded ~~tokens~~, so depth is measured by following
    the actual record values, not by assuming a fixed structure.
    """
    it = iter(records[start:])
    consumed = [0]

    def next_rec():
        try:
            rec = next(it)
            consumed[0] += 1   # only increment when a record is actually returned
            return rec
        except StopIteration:
            return None        # exhausted — do NOT increment consumed

    def consume(value):
        parts = WC_RE.split(str(value))
        for i, p in enumerate(parts):
            if i % 2 == 1:
                rec = next_rec()
                if rec:
                    consume(rec[1])

    parts = WC_RE.split(template)
    for i, p in enumerate(parts):
        if i % 2 == 1:
            rec = next_rec()
            if rec:
                consume(rec[1])

    return consumed[0]


def _find_image_starts(records, n_images, pos_template):
    """
    Find the start index of each image's records in the flat record list.

    Per-image record counts are VARIABLE because sub-wildcards can pick file
    entries with different numbers of embedded ~~tokens~~.  We cannot use a
    fixed stride.  Instead we walk the list image by image, using
    _count_records_for to consume each image's records and advance the cursor.

    Returns a list of (n_images + 1) indices: starts[i] is the first record
    for image i, starts[n_images] is the first record AFTER the positive block
    (i.e. the start of the negative-prompt records, or len(records)).
    """
    if not pos_template or not WC_RE.search(pos_template):
        # No wildcards in template — every image has 0 records
        return [0] * (n_images + 1)

    starts = []
    cursor = 0
    for _ in range(n_images):
        if cursor >= len(records):
            starts.append(cursor)
            continue
        starts.append(cursor)
        stride = _count_records_for(pos_template, records, cursor)
        cursor += stride if stride > 0 else 1  # advance at least 1 to avoid infinite loop

    starts.append(cursor)  # sentinel: start of neg block
    return starts


# ---------------------------------------------------------------------------
# Infotext post-processing -- strip A1111's auto-added quotes
# ---------------------------------------------------------------------------

_CHAIN_QUOTE_RE = re.compile(r'Wildcard chain: "((\\.|[^"])*)"')


def _strip_chain_quotes(infotext):
    """
    A1111's infotext serialiser wraps any value containing a comma or colon
    in double-quotes.  Our wildcard chain will almost always contain both
    (LoRA syntax, comma-separated prompt terms), so it always gets quoted.

    This strips those outer quotes so the chain is stored verbatim for
    easier manual copying from the PNG Info panel.

    Before: Wildcard chain: "~~wc~~<<~~sub~~<<lora:x:1, text>>>>>>"
    After:  Wildcard chain: ~~wc~~<<~~sub~~<<lora:x:1, text>>>>>>
    """
    def replacer(m):
        inner = m.group(1).replace('\\"', '"').replace('\\\\', '\\')
        return f'Wildcard chain: {inner}'
    return _CHAIN_QUOTE_RE.sub(replacer, infotext)


def _patch_create_infotext():
    """
    Wrap modules.processing.create_infotext once so every call has its
    Wildcard chain value de-quoted before the string is returned.
    """
    try:
        import modules.processing as processing
        orig = processing.create_infotext
        if getattr(orig, '_wt_patched', False):
            return  # already patched
        def patched_create_infotext(*args, **kwargs):
            result = orig(*args, **kwargs)
            if isinstance(result, str) and 'Wildcard chain' in result:
                result = _strip_chain_quotes(result)
            return result
        patched_create_infotext._wt_patched = True
        processing.create_infotext = patched_create_infotext
        print("[WildcardTracer] Patched create_infotext (quote stripping)")
    except Exception as e:
        print(f"[WildcardTracer] Could not patch create_infotext: {e}")


# ---------------------------------------------------------------------------
# Script
# ---------------------------------------------------------------------------

class WildcardTracerScript(scripts.Script):

    def title(self):
        return "Wildcard Tracer"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        return []

    def before_process(self, p, *args):
        RECORDER.start()
        raw_prompt = (getattr(p, "prompt", "") or "").strip()
        # Strip any ||WRC||...||/WRC|| sentinel that may still be present if the
        # wildcard-resolver's before_process hasn't fired yet (load-order dependent).
        # We only want the bare prompt template, not the lock chain payload.
        _WRC_S, _WRC_E = "||WRC||", "||/WRC||"
        _cs = raw_prompt.find(_WRC_S)
        if _cs != -1:
            _ce = raw_prompt.find(_WRC_E, _cs)
            raw_prompt = raw_prompt[:_cs] + (raw_prompt[_ce + len(_WRC_E):] if _ce != -1 else "")
        self._wt_pos_template = raw_prompt.strip()
        self._wt_all_records  = None  # populated on first process_before_every_sampling
        self._wt_image_starts = None  # list of per-image start indices into all_records
        self._wt_img_counter  = 0     # per-generation counter, NOT p.iteration
        _patch_create_infotext()
        print(f"[WildcardTracer] before_process -- template={repr(self._wt_pos_template[:60])}")

    def process_before_every_sampling(self, p, *args, **kwargs):
        params = getattr(p, "extra_generation_params", {}) or {}

        # Own counter -- increments once per image regardless of batch structure.
        # p.iteration is the outer batch-count loop index and is NOT reliable
        # as a per-image index into all_prompts.
        img_idx = self._wt_img_counter
        self._wt_img_counter += 1

        # Prefer sd-dynamic-prompts' "Template" param (the raw pre-resolution
        # prompt) when available; fall back to what we captured in before_process.
        # Strip any sentinel that may be embedded if sd-dynamic-prompts set Template
        # before our before_process had a chance to strip it from p.prompt.
        _raw_tpl = (params.get("Template", "") or self._wt_pos_template or "").strip()
        _wrc_s = _raw_tpl.find("||WRC||")
        if _wrc_s != -1:
            _wrc_e = _raw_tpl.find("||/WRC||", _wrc_s)
            _raw_tpl = _raw_tpl[:_wrc_s] + (_raw_tpl[_wrc_e + 8:] if _wrc_e != -1 else "")
        pos_template = _raw_tpl.strip()

        # Resolved positive prompt for this specific image.
        all_prompts = getattr(p, "all_prompts", None)
        if isinstance(all_prompts, list) and all_prompts:
            resolved = (all_prompts[img_idx] if img_idx < len(all_prompts)
                        else all_prompts[0])
        else:
            resolved = getattr(p, "prompt", "") or ""
        resolved = resolved.strip()

        # Snapshot all records exactly once per generation (first call only).
        # sd-dynamic-prompts resolves ALL positive prompts first, then ALL
        # negative prompts, during process() -- before this hook ever fires.
        # Record layout: [img0_pos..., img1_pos..., ..., img0_neg..., img1_neg...]
        #
        # Per-image record counts are VARIABLE: different random picks at lower
        # levels produce different numbers of sub-records.  We cannot use a fixed
        # stride.  Instead we find image boundaries by scanning for the top-level
        # wildcard name repeating in the record list.
        if self._wt_all_records is None:
            self._wt_all_records = RECORDER.snapshot()
            all_prompts_list = getattr(p, "all_prompts", None)
            n_images = len(all_prompts_list) if isinstance(all_prompts_list, list) and all_prompts_list else 1
            self._wt_image_starts = _find_image_starts(
                self._wt_all_records, n_images, pos_template
            )
            print(f"[WildcardTracer] Snapshotted {len(self._wt_all_records)} records, "
                  f"n_images={n_images}, "
                  f"starts={self._wt_image_starts[:min(6,len(self._wt_image_starts))]}")

        all_records   = self._wt_all_records
        image_starts  = self._wt_image_starts

        # Slice this image's records: from its start to the next image's start.
        if img_idx < len(image_starts) - 1:
            start   = image_starts[img_idx]
            end     = image_starts[img_idx + 1]
            records = all_records[start:end]
        else:
            records = []

        print(f"[WildcardTracer] process_before_every_sampling img_idx={img_idx} -- "
              f"template={repr(pos_template[:60])} records={len(records)}")

        if not pos_template or not resolved or pos_template == resolved:
            print("[WildcardTracer] Skipping -- no wildcard resolution detected")
            return

        if not records:
            params["Wildcard chain"] = f"{pos_template}<<{resolved}>>"
            p.extra_generation_params = params
            print("[WildcardTracer] No records -- wrote flat fallback chain")
            return

        chain = build_chain_from_records(pos_template, records)
        if chain:
            params["Wildcard chain"] = chain
            p.extra_generation_params = params
            print(f"[WildcardTracer] Chain written ({len(chain)} chars): {chain[:120]}")

    def postprocess(self, p, processed, *args):
        RECORDER.finish()
        print("[WildcardTracer] postprocess -- records cleared")
