#!/usr/bin/env python3
"""Mid-context retrieval: 3 watchwords at 25/50/75 % of a long filler prompt.
Decode-time attention must *select* those pools (they are far from the tail)."""
import json, random, sys, urllib.request
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 12000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
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
          "\n\nEnd of archive. Three watchwords were embedded in it. "
          "Reply with only the three watchwords, in order, separated by spaces.")
body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": int(sys.argv[3]) if len(sys.argv) > 3 else 3000, "temperature": 0}).encode()
req = urllib.request.Request("http://127.0.0.1:1234/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=1800))
m = r["choices"][0]["message"]; out = m.get("content") or ""; rs = m.get("reasoning") or ""
hits = sum(c in out for c in codes); rhits = sum(c in rs for c in codes)
print(f"target={TARGET} seed={SEED} prompt_tokens={r['usage']['prompt_tokens']} hits={hits}/3 (in reasoning {rhits}/3) gen={r['usage']['completion_tokens']} "
      f"expected={codes} got={out.strip()[:120]!r}", flush=True)
