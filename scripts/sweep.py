"""Parameter sweep: score python CC against the 2013 MatLab outputs (PSNR / SSIM).

usage: python3 scripts/sweep.py OUT.json IMG1,IMG2,... [--workers N]
"""
import argparse
import glob
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize_width  # noqa: E402
from calligraphic_cut import seam_carving as sc  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim  # noqa: E402

REF = os.path.expanduser("~/Projects/ImageResizing2013/drive_raw/Eurographics2013/Our Compare")


def job(args):
    name, cfg = args
    d = os.path.join(REF, name)
    orig = load(os.path.join(d, name + ".png"))
    ref = load(sorted(glob.glob(os.path.join(d, "noline_our*.tif")))[0])
    tw = ref.shape[1]
    t = time.time()
    if cfg == "SC":
        out = sc.resize_width(orig, tw)
    else:
        out = resize_width(orig, tw, Params(**cfg))
    dt = time.time() - t
    tag = "SC" if cfg == "SC" else "_".join(f"{k}{v}" for k, v in cfg.items())
    os.makedirs("out/sweep", exist_ok=True)
    save(f"out/sweep/{name}__{tag}.png", out)
    return dict(name=name, cfg=cfg, tag=tag, psnr=float(psnr(ref, out, data_range=1)),
                ssim=float(ssim(ref, out, channel_axis=2, data_range=1)), sec=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("images")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--grid", default="default")
    a = ap.parse_args()
    cfgs = ["SC"]
    if a.grid == "best":
        cfgs.append(dict(xi1=0.003, xi2=0.3, grad_mode="luma"))
    for x1, x2 in [] if a.grid == "best" else itertools.product([0.001, 0.003, 0.006, 0.012], [0.05, 0.15, 0.3]):
        cfgs.append(dict(xi1=x1, xi2=x2, grad_mode="luma"))
    for x1 in ([] if a.grid == "best" else [0.001, 0.003, 0.006, 0.012]):
        cfgs.append(dict(xi1=x1, xi2=0.15, grad_mode="rgb"))
    jobs = [(n, c) for n in a.images.split(",") for c in cfgs]
    res = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(job, jobs):
            res.append(r)
            print(r["name"], r["tag"], f"psnr {r['psnr']:.2f} ssim {r['ssim']:.3f} {r['sec']:.0f}s", flush=True)
            json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
