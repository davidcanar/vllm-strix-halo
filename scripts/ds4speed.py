#!/usr/bin/env python3
"""DS4 decode speed + coherence probe (thinking off). usage: ds4speed.py [port]"""
import json, sys, time, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "1234"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
def chat(content, max_tokens):
    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
    t0 = time.time()
    with urllib.request.urlopen(urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or "", d["usage"], time.time() - t0
for name, prompt in (("prose", "Explain in detail how a bicycle drivetrain works."),
                     ("json", 'Return a JSON object with keys "name", "age", "city", "hobbies" (a list of 5 strings) and "bio" (two sentences) for a fictional person. Output only JSON.')):
    _, u1, ttft = chat(prompt, 1)
    c, u, dt = chat(prompt, 300)
    n = u["completion_tokens"]
    rate = (n - 1) / max(1e-6, dt - ttft)
    print(f"{name}: {n} tokens, decode {rate:.2f} tok/s ({1000/rate:.0f} ms/token incl. spec), ttft {ttft:.2f}s", flush=True)
    print("   " + repr(c[:300]), flush=True)
