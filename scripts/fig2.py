"""Paper Fig.2: two-colour image whose low-energy regions seams cannot penetrate (512 -> 189)."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize_width  # noqa: E402
from calligraphic_cut import seam_carving as sc  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402

C = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/2013/comparison/")
which = sys.argv[1]
img = load(C + "00OR.png")
t = time.time()
out = resize_width(img, 189, Params()) if which == "cc" else sc.resize_width(img, 189)
os.makedirs("out/fig2", exist_ok=True)
save(f"out/fig2/{which}.png", out)
print(which, f"{time.time() - t:.0f}s")
