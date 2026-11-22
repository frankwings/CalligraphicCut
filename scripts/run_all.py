"""Run CC (numba) + SC on every RetargetMe image that has a 2013 MatLab result.
Writes out/full/<name>_<W>x<H>_{cc,sc}.png and out/full/results.json"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize  # noqa: E402
from calligraphic_cut import seam_carving as sc  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402
from skimage.metrics import structural_similarity as ssim  # noqa: E402

REF = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/Eurographics2013/Our Compare")
OUT = "out/full"


def jobs():
    for d in sorted(os.listdir(REF)):
        for r in sorted(glob.glob(os.path.join(REF, d, "noline_our*.tif"))):
            if os.path.basename(r).startswith("._"):
                continue
            yield d, r


def run(job):
    name, refp = job
    orig = load(os.path.join(REF, name, name + ".png"))
    ref = load(refp)
    H, W = orig.shape[:2]
    h, w = ref.shape[:2]
    tag = f"{name}_{w}x{h}"
    t = time.time()
    cc = resize(orig, target_w=w if w != W else None, target_h=h if h != H else None, p=Params())
    t_cc = time.time() - t
    if h != H:
        s = np.swapaxes(sc.resize_width(np.swapaxes(orig, 0, 1), h), 0, 1)
    else:
        s = sc.resize_width(orig, w)
    save(f"{OUT}/{tag}_cc.png", cc)
    save(f"{OUT}/{tag}_sc.png", s)
    return dict(name=name, tag=tag, ref=refp, orig=(W, H), target=(w, h), sec_cc=t_cc,
                ssim_cc=float(ssim(ref, cc, channel_axis=2, data_range=1)),
                ssim_sc=float(ssim(ref, s, channel_axis=2, data_range=1)))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    res = []
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 16) as ex:
        for r in ex.map(run, list(jobs())):
            res.append(r)
            print(r["tag"], f"cc {r['sec_cc']:.0f}s ssim cc {r['ssim_cc']:.3f} sc {r['ssim_sc']:.3f}", flush=True)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
