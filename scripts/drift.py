import http.client, json, time, threading, glob, statistics as st
def rd(p):
    try: return open(p).read().strip()
    except Exception: return "?"
cpuf = sorted(glob.glob("/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq"))
samples = []
stop = False
def sampler():
    while not stop:
        f = [int(rd(p)) for p in cpuf if rd(p).isdigit()]
        sclk = [l for l in rd("/sys/class/drm/card0/device/pp_dpm_sclk").splitlines() if "*" in l]
        pw = rd("/sys/class/drm/card0/device/hwmon/hwmon*/power1_average") if False else ""
        temps = [rd(p) for p in glob.glob("/sys/class/drm/card0/device/hwmon/hwmon*/temp1_input")]
        samples.append((time.time(), max(f) / 1e6 if f else 0, st.mean(f) / 1e6 if f else 0, sclk[0] if sclk else "?", temps[0] if temps else "?"))
        time.sleep(1)
threading.Thread(target=sampler, daemon=True).start()
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Write a detailed explanation of how a bicycle gear system works."}],
                   "max_tokens": 900, "temperature": 0, "stream": True})
c = http.client.HTTPConnection("localhost", 1234, timeout=900); t0 = time.time()
c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}); r = c.getresponse(); times = []
while True:
    l = r.readline()
    if not l: break
    if l.startswith(b"data:") and b"choices" in l: times.append(time.time())
stop = True
gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
for k in range(0, len(gaps), 25):
    seg = gaps[k:k + 25]; ts = times[k] - t0
    s = [x for x in samples if abs(x[0] - times[k]) < 1.5]
    print(f"t={ts:5.1f}s steps {k:3d}-{k+len(seg):3d} median {st.median(seg):6.1f} ms | cpu max/mean GHz {s[0][1]:.2f}/{s[0][2]:.2f} gpu {s[0][3]} temp {s[0][4]}" if s else f"t={ts:5.1f}s median {st.median(seg):.1f}")
