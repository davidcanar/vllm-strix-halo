#!/usr/bin/env bash
# GLM-5.3 regression battery against the PATCHES 33/34 receipts: warm-up, NLL, decode A/B,
# greedy reproducibility, cold prefill, prefix cache, mid-context needles to 64K, opencode tool-call replay, images.
# usage: scripts/glm_regress.sh [label]    (box1, GLM serving on :1234; ~30 min)
set -u
cd "$(dirname "$0")"
L=${1:-reg}
step() { echo; echo "=== $(date +%T) $*"; }

step "warm-up (the first large prefill after a boot pays Triton JIT)"
t0=$(date +%s)
python3 ttft.py warm 300
python3 ttft.py warm 11500
echo "warm-up took $(( $(date +%s) - t0 )) s"

step "NLL, frozen texts (PATCHES 33, graphs + DFlash2: 2.1948 / 1.8068 / 0.9520)"
python3 nll.py "$L"

step "greedy reproducibility, own cache_salt per rep (PATCHES 15.4: 1 distinct of 5 at ~1K; >2K not re-measured since 27)"
python3 greedy_repro.py 1000 5
python3 greedy_repro.py 6000 5

step "decode A/B (PATCHES 33: 96 / 100 / 98 ms/step, 26.5 / 28.5 / 19.9 tok/s)"
for m in json tools prose; do
    python3 vsh_ab.py --label "$L-$m" --mode "$m" --reps 5 --max-tokens 220
done

step "cold prefill (PATCHES 34: 285-296 tok/s at 4.8K-16.7K)"
for n in 4800 9600 16700; do
    python3 ttft.py cold "$n"
    python3 ttft.py cold "$n"
done

step "prefix cache at 13.9K (PATCHES 20: ~50 s cold -> 0.95 s cached)"
N=REG$(date +%H%M%S)
python3 vsh_bench.py --label "$L-apc-cold" --prompt-tokens 13900 --max-tokens 16 --nonce "$N"
python3 vsh_bench.py --label "$L-apc-warm" --prompt-tokens 13900 --max-tokens 16 --nonce "$N"

step "mid-context needles (3/3 each)"
python3 midneedle.py 6800 7
for s in 7 11 13; do python3 midneedle.py 13000 "$s"; done
python3 midneedle.py 32000 7
python3 midneedle.py 64000 7

step "opencode tool-call replay (8/8)"
python3 replay_n.py 8

step "image probes (inside the container)"
podman exec -e VSH_MODEL=glm-5.3-flash vllm-glm python3 "$PWD/ds4image.py"

step "done"
