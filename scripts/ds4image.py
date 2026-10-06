#!/usr/bin/env python3
"""DS4 image-input probes (Vision-Exp checkpoint, multimodal wrapper): synthetic images with
known content -> OCR, colours/shapes, counting, a table cell, a large wide image.
usage: [VSH_MODEL=glm-5.3-flash] ds4image.py [port]   (needs Pillow; run inside the container)"""
import base64, io, json, os, sys, time, urllib.request
from PIL import Image, ImageDraw, ImageFont

PORT = sys.argv[1] if len(sys.argv) > 1 else "1234"
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("VSH_MODEL", "deepseek-v4-flash")
GLM = MODEL.startswith("glm")

def font(size):
    for f in ("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

def b64(img):
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def ask(img, question, max_tokens=160, thinking=False):
    kw = {"reasoning_effort": "high" if thinking else "low"} if GLM else {"thinking": thinking}
    body = {"model": MODEL, "max_tokens": max(max_tokens, 1200) if GLM else max_tokens, "temperature": 0,
            "chat_template_kwargs": kw,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": b64(img)}},
                                                      {"type": "text", "text": question}]}]}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code}: {e.read().decode()[:300]}", {}, time.time() - t0
    m = d["choices"][0]["message"]
    return (m.get("content") or "") + ((" [reasoning] " + m["reasoning_content"]) if m.get("reasoning_content") and not GLM else ""), d.get("usage", {}), time.time() - t0

def show(tag, ok, ans, usage, dt):
    print(f"{tag:34s} {'PASS' if ok else 'FAIL'}  prompt={usage.get('prompt_tokens', 0):5d} gen={usage.get('completion_tokens', 0):4d} "
          f"{dt:6.1f}s  {ans.strip()[:150]!r}", flush=True)

# 1. text + shapes + colours (768x512)
im = Image.new("RGB", (768, 512), "white"); d = ImageDraw.Draw(im)
d.text((40, 30), "STRIX HALO 7342", fill="black", font=font(64))
d.ellipse((60, 180, 260, 380), fill=(220, 30, 30))
d.rectangle((310, 190, 510, 370), fill=(30, 60, 220))
d.polygon([(640, 180), (540, 380), (740, 380)], fill=(30, 170, 60))
ans, u, dt = ask(im, "What text is written at the top of the image? Then list each shape with its colour.")
a = ans.lower()
show("text + shapes (768x512)", "7342" in a and "red" in a and "blue" in a and "green" in a
     and "circle" in a and ("square" in a or "rectangle" in a) and "triangle" in a, ans, u, dt)

# 2. counting (640x480): 7 yellow discs on dark blue
im = Image.new("RGB", (640, 480), (15, 25, 70)); d = ImageDraw.Draw(im)
for x, y in ((80, 90), (250, 70), (430, 110), (560, 240), (380, 330), (170, 360), (80, 250)):
    d.ellipse((x - 38, y - 38, x + 38, y + 38), fill=(250, 210, 30))
ans, u, dt = ask(im, "How many yellow circles are in the image? Answer with just the number.", 20)
show("count 7 discs (640x480)", "7" in ans or "seven" in ans.lower(), ans, u, dt)

# 3. table cell (900x420): 4x3 grid of numbers
im = Image.new("RGB", (900, 420), "white"); d = ImageDraw.Draw(im); f = font(40)
vals = [["Item", "Qty", "Price"], ["Apples", "12", "3.50"], ["Pears", "7", "4.25"], ["Plums", "31", "9.80"]]
for r, row in enumerate(vals):
    for c, v in enumerate(row):
        x0, y0 = 30 + c * 280, 20 + r * 95
        d.rectangle((x0, y0, x0 + 280, y0 + 95), outline="black", width=3)
        d.text((x0 + 20, y0 + 25), v, fill="black", font=f)
ans, u, dt = ask(im, "In the table, what is the Price of Plums? Answer with just the number.", 20)
show("table cell (900x420)", "9.80" in ans or "9.8" in ans, ans, u, dt)

# 4. large wide image (1920x1080): sign text in a corner, must survive the resize
im = Image.new("RGB", (1920, 1080), (200, 220, 235)); d = ImageDraw.Draw(im)
d.rectangle((1250, 760, 1860, 1020), fill=(0, 110, 50))
d.text((1290, 820), "EXIT 24B", fill="white", font=font(120))
d.ellipse((200, 150, 600, 550), fill=(255, 140, 0))
ans, u, dt = ask(im, "What does the green sign in the bottom-right corner say? Also, what colour is the large circle?", 60)
show("wide 1920x1080: sign + circle", "24b" in ans.lower().replace(" ", "") and "orange" in ans.lower(), ans, u, dt)

# 5. thinking on, same first image
im = Image.new("RGB", (768, 512), "white"); d = ImageDraw.Draw(im)
d.text((40, 30), "STRIX HALO 7342", fill="black", font=font(64))
d.ellipse((60, 180, 260, 380), fill=(220, 30, 30))
ans, u, dt = ask(im, "What number is written in the image?", 400, thinking=True)
show("thinking ON: read the number", "7342" in ans, ans, u, dt)
