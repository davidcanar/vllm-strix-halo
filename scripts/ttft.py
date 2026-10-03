import http.client, json, time, sys, random
random.seed(time.time_ns())   # unique prompt -> no prefix-cache hit
words = "harbor lantern granite meadow cipher kettle ember thistle quarry orchard basalt willow cinder prism".split()
text = " ".join(random.choice(words) for _ in range(int(sys.argv[2]) * 10 // 12))
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": f"[{random.random()}] " + text + "\nSummarize in one word."}], "max_tokens": 1, "temperature": 0})
c = http.client.HTTPConnection("localhost", 1234, timeout=900); t0 = time.time()
c.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"}); u = json.loads(c.getresponse().read())["usage"]; dt = time.time() - t0
print(f"cap={sys.argv[1]} prompt {u['prompt_tokens']} tok  TTFT {dt:.2f} s  -> {u['prompt_tokens']/dt:.0f} tok/s prefill", flush=True)
