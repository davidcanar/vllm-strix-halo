#!/usr/bin/env python3
"""Decode speed deep in a long context: rebuild midneedle.py's prompt (same TARGET/SEED, so the prefix
cache serves it), swap the final question for one with a long answer, and time the stream.
usage: decode_at_ctx.py <target> <seed> [max_tokens]"""
import json, random, sys, time, urllib.request

TARGET, SEED = int(sys.argv[1]), int(sys.argv[2])
MAX_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 400
random.seed(SEED)
words = ("harbor lantern granite meadow cipher kettle ember thistle quarry orchard basalt willow "
         "cinder prism fern mosaic tundra gable spire reef cobalt juniper delta lumen marlin peat "
         "quartz sable tussock vellum walnut zephyr anvil bramble cistern dulse").split()
n_sent = int(TARGET * 4.2 / 6.2 / 9)
hay = [" ".join(random.choice(words) for _ in range(8)) + "." for _ in range(n_sent)]
codes = [f"{random.choice(['AMBER','COBALT','SILVER','IVORY','JADE'])}-{random.choice(['OTTER','FALCON','LYNX','HERON','BISON'])}-{random.randint(1000,9999)}" for _ in range(3)]
for frac, c, tag in zip((0.25, 0.5, 0.75), codes, ("first", "second", "third")):
    hay.insert(int(len(hay) * frac), f"IMPORTANT: the {tag} watchword is {c}.")
prompt = ("Below is a long archive document.\n\n" + " ".join(hay) +
          "\n\nEnd of archive. Write a 300-word story in which all three watchwords appear.")
body = {"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": MAX_TOK,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"reasoning_effort": "low"}}
req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", json.dumps(body).encode(),
                             {"Content-Type": "application/json"})
t0 = time.time(); t_first = None; usage = {}
with urllib.request.urlopen(req, timeout=3600) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices", []):
            delta = ch.get("delta", {})
            if t_first is None and (delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content")):
                t_first = time.time()
t1 = time.time()
gen = usage.get("completion_tokens", 0)
cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
print(f"prompt {usage.get('prompt_tokens')} (cached {cached}) TTFT {t_first - t0:.1f}s  gen {gen} in {t1 - t_first:.1f}s "
      f"-> decode {(gen - 1) / (t1 - t_first):.1f} tok/s", flush=True)
