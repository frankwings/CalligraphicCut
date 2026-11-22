"""Reproduce the paper teaser (Toledo cathedral, 50% / 66% width)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize_width  # noqa: E402
from calligraphic_cut import seam_carving as sc  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402

T = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/2013/teaser/")
which, tw = sys.argv[1], int(sys.argv[2])
img = load(T + "0.jpg")
t = time.time()
out = resize_width(img, tw, Params()) if which == "cc" else sc.resize_width(img, tw)
os.makedirs("out/teaser", exist_ok=True)
save(f"out/teaser/{which}_{tw}.png", out)
print(which, tw, f"{time.time() - t:.0f}s")
