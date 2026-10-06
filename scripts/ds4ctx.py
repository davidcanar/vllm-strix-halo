#!/usr/bin/env python3
"""DS4 long-context probe: three watchwords at 25/50/75 % of a filler document, thinking off,
streamed so prefill (TTFT) and decode speed are measured separately; the answer also counts
1..40 to exercise decode at that context. usage: ds4ctx.py TOKENS[,TOKENS...] [seed] [port]"""
import json, random, sys, time, urllib.request

SIZES = [int(x) for x in sys.argv[1].split(",")]
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
PORT = sys.argv[3] if len(sys.argv) > 3 else "1234"
BASE = f"http://127.0.0.1:{PORT}"
WORDS = ("harbor lantern granite meadow cipher kettle ember thistle quarry orchard basalt willow "
         "cinder prism fern mosaic tundra gable spire reef cobalt juniper delta lumen marlin peat "
         "quartz sable tussock vellum walnut zephyr anvil bramble cistern dulse").split()

def ntok(text):
    req = urllib.request.Request(BASE + "/tokenize", json.dumps({"model": "deepseek-v4-flash", "prompt": text}).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())["count"]

rng = random.Random(SEED)
sample = " ".join(" ".join(rng.choice(WORDS) for _ in range(8)) + "." for _ in range(400))
per_sent = ntok(sample) / 400.0                      # tokens per 8-word sentence

for target in SIZES:
    rng = random.Random(SEED * 1000 + target)
    n = int((target - 120) / per_sent)
    hay = [" ".join(rng.choice(WORDS) for _ in range(8)) + "." for _ in range(n)]
    codes = [f"{rng.choice(['AMBER','COBALT','SILVER','IVORY','JADE'])}-{rng.choice(['OTTER','FALCON','LYNX','HERON','BISON'])}-{rng.randint(1000, 9999)}"
             for _ in range(3)]
    for frac, c, tag in zip((0.25, 0.5, 0.75), codes, ("first", "second", "third")):
        hay.insert(int(len(hay) * frac), f"IMPORTANT: the {tag} watchword is {c}.")
    prompt = ("Below is a long archive document.\n\n" + " ".join(hay) +
              "\n\nEnd of archive. Three watchwords were embedded in it. Reply with the three watchwords, in order, "
              "separated by spaces. Then, on a new line, count from 1 to 40 separated by commas.")
    body = {"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 200,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"thinking": False}}
    t0 = time.time(); first = None; text = ""; usage = {}
    req = urllib.request.Request(BASE + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=4 * 3600) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                if d.get("usage"):
                    usage = d["usage"]
                for ch in d.get("choices", []):
                    piece = ch.get("delta", {}).get("content") or ""
                    if piece and first is None:
                        first = time.time()
                    text += piece
    except Exception as e:  # report and continue with the next size
        print(f"ctx={target}: request failed after {time.time() - t0:.0f}s: {e}", flush=True)
        continue
    t1 = time.time(); first = first or t1
    pt, ct = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
    hits = sum(c in text for c in codes)
    nums = [int(x) for x in __import__("re").findall(r"\b\d+\b", text.split("\n", 1)[-1])]
    good = 0
    for i, v in enumerate(nums):
        if v != i + 1:
            break
        good = i + 1
    ttft = first - t0; dec = (ct - 1) / max(1e-6, t1 - first)
    print(f"ctx={target:7d} prompt_tokens={pt:7d} needles {hits}/3 count-to-40 {'ok' if good >= 40 else f'stopped at {good}'} | "
          f"TTFT {ttft:7.1f}s (prefill {pt / max(ttft, 1e-6):6.1f} tok/s) | decode {dec:5.1f} tok/s ({ct} tokens) | "
          f"{text.strip().splitlines()[0][:70]!r}", flush=True)
