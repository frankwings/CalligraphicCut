"""Per-image comparison montages for the share page."""
import glob
import json
import os

from PIL import Image, ImageDraw

REF = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/Eurographics2013/Our Compare")
OUT = "out/full/montage"
os.makedirs(OUT, exist_ok=True)
res = json.load(open("out/full/results.json"))
H = 360
for r in res:
    name, (W0, H0), (w, h) = r["name"], r["orig"], r["target"]
    ratio = (w / W0) if w != W0 else (h / H0)
    rs = f"{round(ratio * 4) / 4:.2f}"
    panels = [("Original", os.path.join(REF, name, name + ".png")),
              ("CC 2013 MatLab", r["ref"]),
              ("CC Python (new)", f"out/full/{r['tag']}_cc.png"),
              ("SC Python", f"out/full/{r['tag']}_sc.png")]
    for m, lab in [("sc", "SC RetargetMe"), ("sm", "Shift-Map"), ("sv", "Streaming Video")]:
        f = os.path.join(REF, name, f"{name}_{rs}_{m}.png")
        if os.path.exists(f):
            panels.append((lab, f))
    ims = []
    for lab, f in panels:
        im = Image.open(f).convert("RGB")
        ims.append((lab, im.resize((max(1, int(im.width * H / im.height)), H), Image.LANCZOS)))
    Wt = sum(i.width + 8 for _, i in ims)
    c = Image.new("RGB", (Wt, H + 46), "white")
    d = ImageDraw.Draw(c)
    d.text((4, 2), f"{name}: {W0}x{H0} -> {w}x{h}   SSIM vs 2013: CC-py {r['ssim_cc']:.3f} | SC {r['ssim_sc']:.3f}", fill="black")
    x = 0
    for lab, im in ims:
        d.text((x + 3, 22), lab, fill="black")
        c.paste(im, (x, 44))
        x += im.width + 8
    c.save(f"{OUT}/{r['tag']}.jpg", quality=90)
print(len(res))
wins = sum(r["ssim_cc"] > r["ssim_sc"] for r in res)
import numpy as np
print("CC>SC", wins, "/", len(res), "mean cc", np.mean([r["ssim_cc"] for r in res]), "sc", np.mean([r["ssim_sc"] for r in res]))
