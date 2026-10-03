import http.client, json, time, threading, glob, statistics as st, sys
cpuf = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq")); fs = []; stop = False
def samp():
    while not stop:
        v = [int(open(p).read()) for p in cpuf]; fs.append(max(v) / 1e6); time.sleep(0.5)
threading.Thread(target=samp, daemon=True).start()
pad = "Background notes: " + " ".join(f"item {i} is filed under shelf {i*7%13}." for i in range(290))
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": pad + "\n\nExplain how a refrigerator works, in detail. " + sys.argv[1]}],
                   "max_tokens": 450, "temperature": 0, "stream": True, "stream_options": {"include_usage": True}})
c = http.client.HTTPConnection("localhost", 1234, timeout=900); c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
r = c.getresponse(); t = []; u = None
while True:
    l = r.readline()
    if not l: break
    if l.startswith(b"data:") and b'"choices":[{' in l: t.append(time.time())
    if b'"usage"' in l and b"completion_tokens" in l:
        try: u = json.loads(l[5:])["usage"]
        except Exception: pass
stop = True
g = [(b - a) * 1000 for a, b in zip(t, t[1:])][30:]          # steady state only
tps = (u or {}).get("completion_tokens", 0) / max(1, len(t) - 1)
print(f"cap={sys.argv[1]:5s} steady step {st.median(g):6.1f} ms  tok/step {tps:.2f} -> {1000*tps/st.median(g):5.1f} tok/s | cpu max-core GHz median {st.median(fs[10:]):.2f}", flush=True)
