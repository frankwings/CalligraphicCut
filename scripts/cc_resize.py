"""CLI: python3 scripts/cc_resize.py IN OUT [--width W] [--height H] [--xi1 ..] [--xi2 ..] [--viz CUTS.png]"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calligraphic_cut import Params, resize  # noqa: E402
from calligraphic_cut.io import load, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Calligraphic Cutting image resizing")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--xi1", type=float, default=Params.xi1)
    ap.add_argument("--xi2", type=float, default=Params.xi2)
    ap.add_argument("--backend", default="numba", choices=["numba", "numpy"])
    a = ap.parse_args()
    if a.width is None and a.height is None:
        ap.error("give --width and/or --height (shrinking only)")
    img = load(a.inp)
    out = resize(img, a.width, a.height, Params(xi1=a.xi1, xi2=a.xi2, backend=a.backend))
    save(a.out, out)
    print(f"{img.shape[1]}x{img.shape[0]} -> {out.shape[1]}x{out.shape[0]}  {a.out}")


if __name__ == "__main__":
    main()
