"""CLI: cc-resize IN OUT [--width W] [--height H] [--xi1 ..] [--xi2 ..]"""
import argparse

from . import Params, resize
from .io import load, save


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
