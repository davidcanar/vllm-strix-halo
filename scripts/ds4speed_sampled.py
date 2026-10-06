#!/usr/bin/env python3
"""DS4 decode speed with the SERVER's default sampling (no temperature/top_p in the request), thinking off;
median of N runs per prompt (sampled output varies). usage: ds4speed_sampled.py [port] [runs]"""
import json, statistics, sys, time, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "1234"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
def chat(content, max_tokens, thinking=False):
    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "chat_template_kwargs": {"thinking": thinking}}
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        d = json.loads(r.read())
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or m.get("reasoning") or ""), d["usage"], time.time() - t0
for name, prompt in (("prose", "Explain in detail how a bicycle drivetrain works."),
                     ("json", 'Return a JSON object with keys "name", "age", "city", "hobbies" (a list of 5 strings) and "bio" (two sentences) for a fictional person. Output only JSON.')):
    _, _, _, ttft = chat(prompt, 1)
    rates = []
    for _ in range(N):
        c, _, u, dt = chat(prompt, 300)
        rates.append((u["completion_tokens"] - 1) / max(1e-6, dt - ttft))
    print(f"{name}: decode median {statistics.median(rates):.2f} tok/s (runs {' '.join(f'{r:.1f}' for r in rates)}), ttft {ttft:.2f}s | {c[:90]!r}", flush=True)
c, r, u, dt = chat("What is 17 * 23? Answer briefly.", 3000, thinking=True)
print(f"thinking on (server default effort): {u['completion_tokens']} tokens in {dt:.1f}s, reasoning {len(r)} chars, answer {c.strip()[:60]!r}", flush=True)
