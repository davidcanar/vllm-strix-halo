"""Mean per-token NLL of fixed texts (prompt_logprobs) -> quality A/B across builds."""
import json, sys, urllib.request, math
label = sys.argv[1]
texts = {
  "patches": open("/home/davidcanar/vllm-strix-halo/PATCHES.md").read()[2000:9000],
  "agents": open("/home/davidcanar/vllm-strix-halo/AGENTS.md").read()[:7000],
  "pyfile": open("/home/davidcanar/vllm-strix-halo/scripts/vsh_ab.py").read()[:6000],
}
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
