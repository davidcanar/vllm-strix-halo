import json, random, sys, urllib.request
# rebuild the exact midneedle.py prompt and find the watchword pool ids
TARGET, SEED = int(sys.argv[1]), int(sys.argv[2])
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
def ntok(text):
    body = json.dumps({"model": "glm-5.3-flash", "messages": [{"role": "user", "content": text}],
                       "add_generation_prompt": False}).encode()
    r = json.load(urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:1234/tokenize", data=body,
                  headers={"Content-Type": "application/json"})))
    return r["count"]
pools = []
for c in codes:
    i = prompt.index(c)
    s, e = ntok(prompt[:i]), ntok(prompt[: i + len(c)])
    pools += list(range(max(0, s - 2) // 4, e // 4 + 1))
    print(c, "tokens", s, e, "pools", max(0, s - 2) // 4, "-", e // 4)
json.dump(pools, open(sys.argv[3], "w"))
print("total prompt tokens", ntok(prompt), "pools ->", sys.argv[3])
