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
    Dry-run the chain-builder on records[start:] using `template` as the
    structural guide.  Returns the number of records consumed.

    This tells us the per-image record stride: how many records one resolution
    of `template` produces, regardless of what values were actually chosen.
    The structure (number of wildcard slots) is identical for every image
    because the raw template never changes between images.
    """
    it = iter(records[start:])
    consumed = [0]

    def next_rec():
        try:
            consumed[0] += 1
            return next(it)
        except StopIteration:
            return None

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
        self._wt_pos_template = (getattr(p, "prompt", "") or "").strip()
        self._wt_all_records  = None  # populated on first process_before_every_sampling
        self._wt_stride       = None  # records per image -- computed once, reused
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
        pos_template = (params.get("Template", "") or self._wt_pos_template or "").strip()

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
        # The record layout is therefore:
        #
        #   [img0_pos_r0, img0_pos_r1, ...,   <- stride records for image 0
        #    img1_pos_r0, img1_pos_r1, ...,   <- stride records for image 1
        #    ...                              <- more positive images
        #    img0_neg_r0, ...,                <- negative records (ignored)
        #    img1_neg_r0, ...]
        #
        # We only need the positive block.  Because the raw template is
        # identical for every image, every image produces exactly `stride`
        # records.  We compute stride once from the first image's records
        # and reuse it -- no cursor arithmetic, no neg-skipping needed.
        if self._wt_all_records is None:
            self._wt_all_records = RECORDER.snapshot()
            self._wt_stride = _count_records_for(
                pos_template, self._wt_all_records, 0
            )
            print(f"[WildcardTracer] Snapshotted {len(self._wt_all_records)} total records, "
                  f"stride={self._wt_stride}")

        all_records = self._wt_all_records
        stride      = self._wt_stride

        # Slice exactly this image's records: a fixed-width window into the
        # positive block.  Negative records live beyond index N*stride and
        # are never touched.
        start   = img_idx * stride
        records = all_records[start : start + stride]

        print(f"[WildcardTracer] process_before_every_sampling img_idx={img_idx} -- "
              f"template={repr(pos_template[:60])} records={len(records)} "
              f"(slice [{start}:{start+stride}] of {len(all_records)})")
        if records:
            print(f"[WildcardTracer]   record[0]: {records[0][0]!r} -> "
                  f"{repr(str(records[0][1]))[:80]}")

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
