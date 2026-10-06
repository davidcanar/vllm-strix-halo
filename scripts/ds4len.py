#!/usr/bin/env python3
"""DS4 length/position battery: where does output break? usage: ds4len.py [port]"""
import json, re, sys, time, urllib.request
PORT = sys.argv[1] if len(sys.argv) > 1 else "1234"
BASE = f"http://127.0.0.1:{PORT}/v1"
MODEL = "deepseek-v4-flash"
SENTS = ["The river valley was settled by farmers who grew barley and kept goats.",
         "Over the centuries the town built a stone bridge, a granary and a small library.",
         "Merchants travelled the northern road each spring, trading wool for salt and iron.",
         "In the archives a clerk recorded every harvest, flood and festival in careful ink.",
         "The old mill still turns when the snow melts and the water runs fast and cold.",
         "Children learned their letters from a teacher who also repaired clocks.",
         "A lighthouse on the far headland guided fishing boats home through the fog.",
         "Each autumn the families gathered apples and pressed them into cider."]

def post(path, body, timeout=900):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read())
    return d, time.time() - t0

def passage(n_sent):
    return " ".join(f"({i+1}) {SENTS[i % len(SENTS)]}" for i in range(n_sent))

def chat(content, max_tokens, thinking):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens,
            "temperature": 0, "chat_template_kwargs": {"thinking": thinking}}
    d, dt = post("/chat/completions", body)
    m = d["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or m.get("reasoning") or ""), d["usage"], dt

def comp(prompt, max_tokens):
    d, dt = post("/completions", {"model": MODEL, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0})
    return d["choices"][0]["text"], d["usage"], dt

def show(tag, ok, usage, dt, text):
    print(f"{tag:42s} {'PASS' if ok else 'FAIL'}  prompt={usage['prompt_tokens']:5d} gen={usage['completion_tokens']:4d} {dt:6.1f}s  {text[:90]!r}", flush=True)

Q = "\n\nQuestion: what is the capital of France? Answer with one word."
# A/B: chat, thinking off, short vs long prompts
for ns in (0, 4, 12, 24, 60):
    c, r, u, dt = chat((passage(ns) + Q) if ns else "What is the capital of France? Answer with one word.", 16, False)
    show(f"chat think=off passage={ns} sentences", "Paris" in c, u, dt, c)
# C: long generation crossing 128 positions
c, r, u, dt = chat("Count from 1 to 100, separated by commas. Output only the numbers.", 420, False)
nums = [int(x) for x in re.findall(r"\d+", c)]
good = 0
for i, v in enumerate(nums):
    if v != i + 1: break
    good = i + 1
show("chat think=off count 1..100", good >= 100, u, dt, c[-80:])
print(f"   counted correctly up to {good}; decode {u['completion_tokens']/dt:.2f} tok/s (incl. prefill)", flush=True)
# D/G: raw completions sweep across ~128 positions
for ns in (1, 4, 6, 8, 9, 10, 12, 16, 30):
    t, u, dt = comp(passage(ns) + "\nThe capital of France is", 8)
    show(f"raw completion passage={ns} sentences", "Paris" in t, u, dt, t)
# H: needle 8K, thinking off
filler = passage(380)
needle = filler[: len(filler) // 2] + " The secret code is AMBER-FALCON-4455. " + filler[len(filler) // 2:]
c, r, u, dt = chat(needle + "\n\nWhat is the secret code? Answer with only the code.", 24, False)
show("chat think=off needle ~8K", "AMBER-FALCON-4455" in c, u, dt, c)
# F: thinking on (reproduce)
c, r, u, dt = chat("What is the capital of France? Answer with one word.", 200, True)
show("chat think=ON short", "Paris" in (c + r), u, dt, (r[:50] + " || " + c[:40]))
