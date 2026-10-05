#!/usr/bin/env python3
"""DS4 Phase-0 probe battery. usage: ds4probe.py [mode]
modes: quick (coherence+counting+determinism+speed) | needle8K | needle32K | needle <toks>"""
import json, sys, time, hashlib, urllib.request

URL = "http://127.0.0.1:1234/v1/chat/completions"
MODEL = "deepseek-v4-flash"
MAGIC = "XZYQUUX"

def send(msgs, max_tokens=120, temperature=0):
    body = {"model": MODEL, "messages": msgs, "max_tokens": max_tokens, "temperature": temperature}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    m = d["choices"][0]["message"]
    u = d["usage"]
    return (m.get("content") or "", dt, u["prompt_tokens"], u["completion_tokens"])

def filler(target_tokens, salt):
    words = (f"grant {salt} ledger quiet bridge lantern mosaic drift ",)
    line = f"journal {salt} entry {salt} ledger quiet bridge lantern mosaic driftSignal pager byte fork amber cinder plume gravel token "
    out = []
    n = 0
    i = 0
    while n < target_tokens:
        s = f"{line}pass {i} {salt} "
        out.append(s)
        n += len(s) // 4
        i += 1
    return "".join(out), n

def needle(total_tokens, depth):
    salt = hashlib.md5(f"{total_tokens}".encode()).hexdigest()[:8]
    text, n = filler(total_tokens, salt)
    pos = int(len(text) * depth)
    text = text[:pos] + f" The magic word is {MAGIC}. " + text[pos:]
    return [{"role": "user", "content": text + "\n\nWhat is the magic word? Answer with only the word."}], n + 8

mode = sys.argv[1] if len(sys.argv) > 1 else "quick"

if mode == "quick":
    c, dt, pt, ct = send([{"role": "user", "content": "What is the capital of France? One word."}], 16)
    print(f"capital: {dt=:.2f}s out={c[:40]!r} ({'PASS' if 'Paris' in c else 'FAIL'})")
    c2, dt2, pt2, ct2 = send([{"role": "user", "content": "Count from one to nineteen."}], 80)
    print(f"count: {dt2=:.2f}s {ct2}tok {ct2/dt2:.1f}tok/s out={c2[:80]!r}")
    c3, dt3, *_ = send([{"role": "user", "content": "What is the capital of France? One word."}], 16)
    print(f"determinism: {'PASS' if c == c3 else 'FAIL'} ({c[:20]!r} vs {c3[:20]!r})")
    c4, dt4, pt4, ct4 = send([{"role": "user", "content": "Write three varied sentences about the sea."}], 96)
    print(f"prose: {dt4=:.2f}s {ct4/dt4:.1f}tok/s out={c4[:120]!r}")

elif mode.startswith("needle"):
    toks = int(mode[6:]) if len(mode) > 6 else (int(sys.argv[2]) if len(sys.argv) > 2 else 8000)
    for depth in (0.5, 0.9):
        msgs, n = needle(toks, depth)
        c, dt, pt, ct = send(msgs, 24)
        ok = "PASS" if MAGIC in c else "FAIL"
        print(f"needle {toks}K? tokens={pt} depth={depth}: {dt=:.1f}s out={c[:40]!r} -> {ok}")
