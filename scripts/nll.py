"""Mean per-token NLL of fixed texts (prompt_logprobs) -> quality A/B across builds."""
import json, sys, urllib.request, math
label = sys.argv[1]
# Frozen copies of PATCHES.md[2000:9000], AGENTS.md[:7000] and scripts/vsh_ab.py[:6000] as
# scored for the 2026-10-03 baselines (PATCHES 33/34), so editing those files cannot move the NLL.
_D = "/home/davidcanar/vllm-strix-halo/scripts/data"
texts = {name: open(f"{_D}/nll_{name}.txt", encoding="utf-8").read() for name in ("patches", "agents", "pyfile")}
out = {}
for name, t in texts.items():
    body = json.dumps({"model": "glm-5.3-flash", "prompt": t, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:1234/v1/completions", data=body, headers={"Content-Type": "application/json"}), timeout=900))
    pl = r["choices"][0].get("prompt_logprobs") or []
    lps = [list(d.values())[0]["logprob"] for d in pl if d]
    nll = -sum(lps) / len(lps)
    out[name] = (len(lps), nll)
    print(f"[{label}] {name:8s} tokens {len(lps):5d}  mean NLL {nll:.4f}  ppl {math.exp(nll):.3f}", flush=True)
json.dump(out, open(f"/tmp/nll_{label}.json", "w"))
