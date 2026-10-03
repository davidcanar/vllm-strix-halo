#!/usr/bin/env python3
"""Unprofiled decode step time: stream a request, time each SSE chunk (one chunk ~= one engine step).
usage: stepbench.py LABEL [reps]  -> prints median step ms + tok/step for short (24 tok) and ~3K contexts."""
import http.client, json, sys, time, statistics as st
LABEL = sys.argv[1]; REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
def run(prompt, ntok=200):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": ntok, "temperature": 0, "stream": True,
                       "stream_options": {"include_usage": True}})
    c = http.client.HTTPConnection("localhost", 1234, timeout=900)
    c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
    r = c.getresponse(); times = []; usage = None
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]": continue
        j = json.loads(line[5:])
        if j.get("usage"): usage = j["usage"]
        if j.get("choices"): times.append(time.time())
    gaps = [(b - a) * 1000 for a, b in zip(times[2:], times[3:])]   # skip first steps after prefill
    return st.median(gaps), usage["completion_tokens"] / max(1, len(times) - 1)
pad = "Background notes: " + " ".join(f"item {i} is filed under shelf {i*7%13}." for i in range(290))
res = {}
for name, pr in (("short", "Write a detailed explanation of how a bicycle gear system works."),
                 ("3k", pad + "\n\nExplain how a refrigerator works, in detail.")):
    ms, tps = zip(*(run(pr if k == 0 else pr + f" (variant {k})") for k in range(REPS)))
    res[name] = (st.median(ms), st.median(tps))
    print(f"[{LABEL}] {name:5s} step {st.median(ms):6.1f} ms (runs {', '.join('%.0f' % m for m in ms)})  "
          f"tok/step {st.median(tps):.2f}  -> {1000 * st.median(tps) / st.median(ms):.1f} tok/s", flush=True)
