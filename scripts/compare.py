"""Run CC (new python) + SC baseline on a RetargetMe image and montage against the 2013 MatLab result.

usage: python3 scripts/compare.py NAME [--xi1 ..] [--xi2 ..] [--tag ..]
"""
import argparse
import glob
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize_width  # noqa: E402
from calligraphic_cut import seam_carving as sc  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402

REF = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/Eurographics2013/Our Compare")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("--xi1", type=float, default=Params.xi1)
    ap.add_argument("--xi2", type=float, default=Params.xi2)
    ap.add_argument("--grad", default="luma")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-sc", action="store_true")
    a = ap.parse_args()
    d = os.path.join(REF, a.name)
    orig = load(os.path.join(d, a.name + ".png"))
    ref_path = sorted(glob.glob(os.path.join(d, "noline_our*.tif")))[0]
    ref = load(ref_path)
    tw = ref.shape[1]
    os.makedirs("out", exist_ok=True)
    p = Params(xi1=a.xi1, xi2=a.xi2, grad_mode=a.grad)
    t = time.time()
    cc = resize_width(orig, tw, p)
    t_cc = time.time() - t
    tag = a.tag or f"x1_{a.xi1}_x2_{a.xi2}_{a.grad}"
    save(f"out/{a.name}_cc_{tag}.png", cc)
    panels = [("original", orig), ("CC 2013 MatLab", ref), (f"CC python ({tag})", cc)]
    if not a.no_sc:
        scp = f"out/{a.name}_sc.png"
        if os.path.exists(scp):
            s = load(scp)
        else:
            s = sc.resize_width(orig, tw)
            save(scp, s)
        panels.append(("SC python", s))
    H = orig.shape[0]
    W = sum(x.shape[1] for _, x in panels) + 10 * len(panels)
    canvas = Image.new("RGB", (W, H + 24), "white")
    dr = ImageDraw.Draw(canvas)
    x = 0
    for lab, im in panels:
        canvas.paste(Image.fromarray(np.clip(np.rint(im * 255), 0, 255).astype(np.uint8)), (x, 24))
        dr.text((x + 4, 4), lab, fill="black")
        x += im.shape[1] + 10
    canvas.save(f"out/{a.name}_cmp_{tag}.png")
    print(f"{a.name}: {orig.shape[1]}->{tw}  cc {t_cc:.1f}s  -> out/{a.name}_cmp_{tag}.png")


if __name__ == "__main__":
    main()
