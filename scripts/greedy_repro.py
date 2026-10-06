#!/usr/bin/env python3
"""Greedy reproducibility: N byte-identical temperature-0 requests, each with its own cache_salt
so the prefix cache cannot serve the repeats, -> how many distinct completions.
Baselines: PENDINGWORK 1.1 (5 distinct at every length), PATCHES 15.4 (1 distinct at ~1K tokens,
still 5 above ~2K); PATCHES 27 suspected the >2K part was its sparse-attention bug family.
usage: greedy_repro.py <prompt_tokens> [reps] [max_tokens] [port]"""
import hashlib, json, random, sys, time, urllib.request

N_TOK = int(sys.argv[1])
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 5
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 96
PORT = sys.argv[4] if len(sys.argv) > 4 else "1234"
rng = random.Random(1234)
words = ("harbor lantern granite meadow cipher kettle ember thistle quarry orchard basalt willow "
         "cinder prism fern mosaic tundra gable spire reef cobalt juniper delta lumen").split()
text = " ".join(rng.choice(words) for _ in range(int(N_TOK * 0.8)))
prompt = (f"Archive:\n{text}\n\nWrite a short story that uses the five words that appear most "
          "often in the archive.")
tag = time.strftime("%H%M%S")
outs = []
for i in range(REPS):
    body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOK, "temperature": 0, "seed": 1234,
            "chat_template_kwargs": {"reasoning_effort": "low"}, "cache_salt": f"repro-{tag}-{i}"}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    r = json.load(urllib.request.urlopen(req, timeout=1800))
    m = r["choices"][0]["message"]
    t = (m.get("reasoning") or m.get("reasoning_content") or "") + "\x00" + (m.get("content") or "")
    outs.append(t)
    u = r["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"rep {i}: prompt {u['prompt_tokens']} (cached {cached}) gen {u['completion_tokens']} "
          f"{time.time() - t0:5.1f}s md5 {hashlib.md5(t.encode()).hexdigest()[:8]}", flush=True)
print(f"greedy_repro prompt~{u['prompt_tokens']} tokens: {len(set(outs))} distinct / {REPS}", flush=True)
