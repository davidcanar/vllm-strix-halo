#!/usr/bin/env python3
"""vsh-adaptive-k v2 -> v3 install (2026-10-02).

Fixes the v1 port so the EMA policy can actually act on this build:

  * v1 trimmed `request.spec_token_ids` inside
    `Scheduler.update_draft_token_ids`, which this build never calls --
    async scheduling is resolved ON for spec decode and
    `vllm/v1/engine/core.py:252` skips the sync update entirely.  In async
    mode the *verify length* is fixed at schedule time by
    `SchedulerOutput.num_spec_tokens_to_schedule`:
    `AsyncScheduler._update_after_schedule` sizes the placeholder spec
    tokens from it and the worker hands it straight to the drafter
    (`gpu_model_runner.py:4972` -> `drafter.propose(num_speculative_tokens=)`).
    So the cap inside `schedule()` is the live lever; the sync-path trim is
    kept only for the non-async configuration.

  * v1 had no way to observe whether the cap fired.  v2 adds per-step
    telemetry (`VSH_ADAPTIVE_K_DEBUG=1`): the k that was *scheduled*, the k
    that was actually *verified* (from `scheduled_spec_decode_tokens`), the
    per-request choices and the accepted-token histogram.

  * v2 adds a live override file (`VSH_ADAPTIVE_K_JSON`, default
    ~/vsh-adaptive-k.json): `{"force": 2}` pins the batch verify length to 2
    on the next step, `{"force": 0}` releases it, and `{"margin": 0.0}`
    retunes the policy -- so a k-sweep needs one boot, not four.

Idempotent and fail-closed: every anchor must match exactly once, the file is
backed up first, and the result must compile.
"""
from __future__ import annotations

import pathlib
import py_compile
import shutil
import time

P = pathlib.Path(
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
)
MARK = "# [vsh-adaptive-k]"
V2 = "[vsh-adaptive-k] v2"

# ---------------------------------------------------------------- new helper
HELPER_V1_HEAD = "class _VshAdaptiveK:  # [vsh-adaptive-k]"
HELPER_V1_TAIL = "_VSH_ADAPTIVE_K = _VshAdaptiveK()  # [vsh-adaptive-k]"

HELPER_V2 = '''class _VshAdaptiveK:  # [vsh-adaptive-k]
    """CPU-only EMA policy for the verified draft-prefix length (MTP k<=K).

    v2 2026-10-02 -- why the v1 port could not fire on this build:

    * `Scheduler.update_draft_token_ids` (where v1 trimmed) is never called
      here: async scheduling is ON, and `vllm/v1/engine/core.py:252` guards it
      with `not self.async_scheduling`.
    * In async mode the verify length for the next step is fixed at schedule
      time by `SchedulerOutput.num_spec_tokens_to_schedule`:
      `AsyncScheduler._update_after_schedule` sizes `_spec_token_placeholders`
      from it, and the worker passes it to the drafter
      (`gpu_model_runner.py:4972` -> `drafter.propose(num_speculative_tokens=)`).
      The cap inside `schedule()` is therefore the live lever.
    * The sync-path trim is kept for the `async_scheduling=False` case.
    """

    def __init__(self) -> None:
        def _e(name, default):
            v = os.environ.get(name)
            return default if v is None or not str(v).strip() else str(v).strip()

        mode = _e("VSH_ADAPTIVE_K", "off").lower()
        self.boot_enabled = mode in ("ema", "on", "1")
        self.enabled = self.boot_enabled
        self.alpha = float(_e("VSH_ADAPTIVE_K_ALPHA", "0.25"))
        self.margin = float(_e("VSH_ADAPTIVE_K_MARGIN", "1.0"))
        self.min_steps = int(_e("VSH_ADAPTIVE_K_MIN_STEPS", "4"))
        raw = _e("VSH_ADAPTIVE_K_SET", "1,2,3")
        self.k_set = sorted({int(x) for x in raw.split(",") if x.strip()})
        self.saturate = _e("VSH_ADAPTIVE_K_SATURATE", "max").lower()
        self.hist_every = int(_e("VSH_ADAPTIVE_K_HIST", "200"))
        self.debug = _e("VSH_ADAPTIVE_K_DEBUG", "0").lower() in ("1", "on", "true")
        self.debug_every = int(_e("VSH_ADAPTIVE_K_DEBUG_EVERY", "50"))
        self.json_path = _e("VSH_ADAPTIVE_K_JSON", "")
        self.force_k = int(_e("VSH_ADAPTIVE_K_FORCE", "0"))
        self.k_max = max(self.k_set) if self.k_set else 0
        self.state = {}
        self.hist = {}
        self.steps = 0
        self.hist_sched = {}
        self.hist_verified = {}
        self.hist_choice = {}
        self.hist_accepted = {}
        self._json_cache = None
        self._json_mtime = None
        self._json_note = None
        if self.enabled or self.force_k:
            print(
                f"[vsh-adaptive-k] enabled={self.enabled} set={self.k_set} "
                f"alpha={self.alpha} margin={self.margin} min_steps={self.min_steps} "
                f"saturate={self.saturate} force={self.force_k} "
                f"json={self.json_path or '-'} debug={int(self.debug)}",
                flush=True,
            )

    # ------------------------------------------------ live override (no restart)
    def _json(self):
        if not self.json_path:
            return None
        try:
            st = os.stat(self.json_path)
        except OSError:
            return self._json_cache
        if self._json_mtime != st.st_mtime:
            try:
                with open(self.json_path) as fh:
                    self._json_cache = json.load(fh)
            except Exception as exc:  # noqa: BLE001 - never break the step loop
                self._json_cache = None
                print(f"[vsh-adaptive-k] override unreadable: {exc}", flush=True)
            self._json_mtime = st.st_mtime
            print(f"[vsh-adaptive-k] override -> {self._json_cache}", flush=True)
        return self._json_cache

    def force(self):
        """Batch-wide verify length to pin, or None to let the EMA decide."""
        if self.force_k:
            return self.force_k
        j = self._json()
        if isinstance(j, dict):
            for key in ("margin", "alpha"):
                if key in j:
                    try:
                        setattr(self, key, float(j[key]))
                    except (TypeError, ValueError):
                        pass
            if "force" in j:
                try:
                    return int(j["force"]) or None
                except (TypeError, ValueError):
                    return None
        return None

    # ------------------------------------------------------------- observation
    def observe(self, req_id, num_draft, num_accepted) -> None:
        if not (self.enabled or self.force_k) or num_draft <= 0:
            return
        if num_accepted >= num_draft and self.saturate != "n":
            obs = float(max(self.k_max, num_draft))
        else:
            obs = float(num_accepted)
        st = self.state.get(req_id)
        if st is None:
            self.state[req_id] = [
                obs * self.alpha + float(self.k_max) * (1.0 - self.alpha),
                1.0,
            ]
        else:
            st[0] = obs * self.alpha + st[0] * (1.0 - self.alpha)
            st[1] += 1.0
        key = num_accepted if num_accepted < 4 else 3
        self.hist_accepted[key] = self.hist_accepted.get(key, 0) + 1

    # ---------------------------------------------------------------- decision
    def choose(self, req_id, k, structured):
        """Draft length for one request, or None when it must stay at full length."""
        if not self.enabled or structured or k <= 0:
            return None
        st = self.state.get(req_id)
        if st is None or st[1] < self.min_steps:
            return None
        import math

        target = int(math.ceil(st[0] + self.margin))
        cands = [v for v in self.k_set if v <= min(target, k)]
        n = max(cands) if cands else min(self.k_set)
        n = max(1, min(n, k))
        self.hist_choice[n] = self.hist_choice.get(n, 0) + 1
        return n

    # --------------------------------------------------------------- telemetry
    def note_scheduled(self, k) -> None:
        self.hist_sched[k] = self.hist_sched.get(k, 0) + 1

    def note_verified(self, n) -> None:
        self.hist_verified[n] = self.hist_verified.get(n, 0) + 1

    @staticmethod
    def _fmt(d) -> str:
        return " ".join(f"{k}:{v}" for k, v in sorted(d.items())) or "-"

    def tick(self, live_ids=None) -> None:
        """Once per engine step: the periodic receipt and the legacy histogram."""
        self.steps += 1
        if self.hist_every > 0 and self.steps % self.hist_every == 0 and self.hist:
            total = sum(self.hist.values())
            if not self.debug:
                parts = " ".join(f"{k}:{v}" for k, v in sorted(self.hist.items()))
                print(
                    f"[vsh-adaptive-k] step {self.steps} chosen-length hist "
                    f"({total}): {parts}",
                    flush=True,
                )
        if self.debug and self.debug_every > 0 and self.steps % self.debug_every == 0:
            ema = sorted(round(v[0], 2) for v in self.state.values())
            print(
                f"[vsh-adaptive-k] step={self.steps} reqs={len(self.state)} "
                f"sched_k({self._fmt(self.hist_sched)}) "
                f"verified_k({self._fmt(self.hist_verified)}) "
                f"chosen({self._fmt(self.hist_choice)}) "
                f"accepted({self._fmt(self.hist_accepted)}) "
                f"force={self.force()} ema[:6]={ema[:6]}",
                flush=True,
            )
        if self.steps % (self.debug_every * 20 if self.debug else self.hist_every * 20) == 0:
            self.hist_sched = {}
            self.hist_verified = {}
            self.hist_choice = {}
            self.hist_accepted = {}
            if live_ids is not None:
                live = set(live_ids)
                self.state = {r: s for r, s in self.state.items() if r in live}

    # --------------------------------------- sync path (async_scheduling=False)
    def apply(self, reqs, live_ids) -> None:
        """Trim spec_token_ids to a uniform n across the batch (sync path)."""
        if not self.enabled or not reqs:
            return
        ns = []
        for r, s in reqs:
            n_i = self.choose(r.request_id, len(r.spec_token_ids), s)
            if n_i is None:
                ns = None
                break
            ns.append(n_i)
        n = max(len(r.spec_token_ids) for r, _ in reqs) if ns is None else min(ns)
        for r, _ in reqs:
            if len(r.spec_token_ids) > n:
                r.spec_token_ids = r.spec_token_ids[:n]
        self.hist[n] = self.hist.get(n, 0) + 1


_VSH_ADAPTIVE_K = _VshAdaptiveK()  # [vsh-adaptive-k]'''

# ------------------------------------------------------------------ new hooks
OBS_V1 = (
    "                num_accepted = max(len(generated_token_ids) - num_sampled, 0)\n"
    f"                _VSH_ADAPTIVE_K.observe(req_id, num_draft_tokens, num_accepted)  {MARK}\n"
)
OBS_V2 = (
    "                num_accepted = max(len(generated_token_ids) - num_sampled, 0)\n"
    f"                _VSH_ADAPTIVE_K.observe(req_id, num_draft_tokens, num_accepted)  {MARK}\n"
    f"                _VSH_ADAPTIVE_K.note_verified(num_draft_tokens)  {MARK}\n"
)

SCHED_V1 = f"""        if _VSH_ADAPTIVE_K.boot_enabled and self.dynamic_sd_lookup is None and num_scheduled_tokens:  {MARK}
            _ak_ns = []
            for _rid in num_scheduled_tokens:
                _r = self.requests.get(_rid)
                if _r is None or getattr(_r, "is_prefill_chunk", False):
                    continue
                _n_i = _VSH_ADAPTIVE_K.choose(_rid, self.num_spec_tokens, getattr(_r, "structured_output_request", None) is not None)
                if _n_i is None:
                    _ak_ns = None
                    break
                _ak_ns.append(_n_i)
            if _ak_ns:
                num_spec_tokens_to_schedule = min(min(_ak_ns), num_spec_tokens_to_schedule)
"""
SCHED_V2 = f"""        if _VSH_ADAPTIVE_K.boot_enabled or _VSH_ADAPTIVE_K.force_k:  {MARK}
            _ak_force = _VSH_ADAPTIVE_K.force()
            if _ak_force:
                # Live override: pin the whole batch (k-sweep without a restart).
                num_spec_tokens_to_schedule = min(_ak_force, num_spec_tokens_to_schedule)
            elif _VSH_ADAPTIVE_K.boot_enabled and self.dynamic_sd_lookup is None and num_scheduled_tokens:
                _ak_ns = []
                for _rid in num_scheduled_tokens:
                    _r = self.requests.get(_rid)
                    if _r is None or getattr(_r, "is_prefill_chunk", False):
                        continue
                    _n_i = _VSH_ADAPTIVE_K.choose(_rid, self.num_spec_tokens, getattr(_r, "structured_output_request", None) is not None)
                    if _n_i is None:
                        _ak_ns = None
                        break
                    _ak_ns.append(_n_i)
                if _ak_ns:
                    num_spec_tokens_to_schedule = min(min(_ak_ns), num_spec_tokens_to_schedule)
            _VSH_ADAPTIVE_K.note_scheduled(num_spec_tokens_to_schedule)
"""


def main() -> int:
    text = P.read_text()
    if V2 in text:
        print("already at v2 -- nothing to do")
        return 0
    if text.count(MARK) == 0:
        raise SystemExit("no vsh-adaptive-k marker: v1 not installed? refusing")

    backup = P.with_suffix(f".py.bak-{time.strftime('%Y%m%d-%H%M%S')}")
    shutil.copy2(P, backup)

    # 1) imports: json next to the os that v1 added (or before `import time`)
    if "\nimport json\n" not in text:
        for anchor in ("import itertools\nimport os\nimport time\n",):
            if text.count(anchor) == 1:
                text = text.replace(
                    anchor, "import itertools\nimport json\nimport os\nimport time\n", 1
                )
                break
        else:
            raise SystemExit("no safe `import json` anchor")

    # 2) helper class: replace the whole v1 block (head..tail inclusive)
    i = text.find(HELPER_V1_HEAD)
    j = text.find(HELPER_V1_TAIL)
    if i == -1 or j == -1 or j < i:
        raise SystemExit("v1 helper block not found")
    text = text[:i] + HELPER_V2 + text[j + len(HELPER_V1_TAIL) :]

    # 3) OBS site: add the verified-k receipt
    if text.count(OBS_V1) != 1:
        raise SystemExit(f"OBS anchor count {text.count(OBS_V1)}")
    text = text.replace(OBS_V1, OBS_V2, 1)

    # 4) schedule() cap: live override + note_scheduled
    if text.count(SCHED_V1) != 1:
        raise SystemExit(f"SCHED anchor count {text.count(SCHED_V1)}")
    text = text.replace(SCHED_V1, SCHED_V2, 1)

    # 5) tick once per engine step (the signature spans several lines, so the
    #    insertion point is the line that closes it: "    ) -> ...:").
    src = text.splitlines(keepends=True)
    hit = None
    for n, line in enumerate(src):
        if line.startswith("    def update_from_output("):
            hit = n
            break
    if hit is None:
        raise SystemExit("update_from_output not found")
    close = None
    for m in range(hit + 1, min(hit + 12, len(src))):
        if src[m].startswith("    ) ->"):
            close = m
            break
    if close is None:
        raise SystemExit("could not find the end of the update_from_output signature")
    src.insert(close + 1, f"        _VSH_ADAPTIVE_K.tick(self.requests)  {MARK}\n")
    text = "".join(src)
    if "_VSH_ADAPTIVE_K.tick(self.requests)  # [vsh-adaptive-k]" not in text:
        raise SystemExit("tick insertion vanished")

    compile(text, str(P), "exec")
    P.write_text(text)
    py_compile.compile(str(P), doraise=True, cfile="/tmp/_ak_v2.pyc")
    print(f"adaptive-k v2 installed; backup={backup.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
