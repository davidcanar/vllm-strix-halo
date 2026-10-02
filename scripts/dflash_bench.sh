#!/usr/bin/env bash
# DFlash2 k=7: per-position acceptance curve + throughput, next to the MTP k=3
# baseline (2.00-2.06 accepted tokens/step, 8.5-8.9 tok/s on this prompt).
set -u
cd "$HOME"
OUT=$HOME/dflash.txt
: > "$OUT"
note() { echo "=================== $* ===================" | tee -a "$OUT"; }

note "receipts: drafter + attention backend"
journalctl --user -u vsh-glm-manual --no-pager -o cat --since "-12min" 2>/dev/null \
  | grep -aiE "DFlash2|dflash|Draft model|aux layers|attention backend|Using .*backend" | tail -8 | tee -a "$OUT"

note "decode: DFlash2 k=7 (400 tokens, prose)"
for i in 1 2; do
  timeout 900 python3 vsh_bench.py --label "dflash$i" --mode prose --prompt-tokens 2000 \
      --max-tokens 400 --nonce BEEF01 --salt ksw1 2>&1 | tee -a "$OUT"
done

note "decode: structured output (tool-call shaped)"
timeout 900 python3 vsh_bench.py --label dflash-struct --mode structured --prompt-tokens 2000 \
    --max-tokens 200 --nonce BEEF01 --salt ksw2 2>&1 | grep -E "TTFT|spec:|prefix" | tee -a "$OUT"
echo DONE | tee -a "$OUT"
