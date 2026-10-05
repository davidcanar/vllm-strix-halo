#!/usr/bin/env python3
import json, time, urllib.request
body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": "Count from one to thirty."}],
        "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"thinking": False}}
req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions",
                             data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
t0 = time.time()
with urllib.request.urlopen(req, timeout=600) as r:
    d = json.loads(r.read())
u = d["usage"]
print(f"gen wall: {time.time()-t0:.1f}s, completion={u['completion_tokens']}, {u['completion_tokens']/(time.time()-t0):.1f} tok/s")
