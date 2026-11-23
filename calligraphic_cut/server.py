"""Local experiment server (Gradio).

    cc-server                       # http://127.0.0.1:8097
    cc-server --port 9000 --host 0.0.0.0
    CC_ROOT_PATH=/cc cc-server      # when served behind a reverse proxy / Tailscale Funnel path
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from . import seam_carving as sc
from .core import Params, best_cut, resize

MAX_SIDE = 1600
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "tests" / "data" / "paper"


# ---------------------------------------------------------------- helpers
def _to_float(img: np.ndarray):
    note = ""
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=2)
    img = img[..., :3]
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = np.asarray(Image.fromarray(img).resize((round(w * s), round(h * s)), Image.LANCZOS))
        note = f" (input downscaled from {w}x{h} to {img.shape[1]}x{img.shape[0]})"
    return np.ascontiguousarray(img, dtype=np.float64) / 255.0, note


def _to_u8(img):
    return np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)


def _targets(img, wpct, hpct):
    h, w = img.shape[:2]
    tw = max(2, round(w * wpct / 100))
    th = max(2, round(h * hpct / 100))
    return (tw if tw < w else None), (th if th < h else None)


def _sc_resize(img, tw, th):
    out = img
    if tw:
        out = sc.resize_width(out, tw)
    if th:
        out = np.swapaxes(sc.resize_width(np.swapaxes(out, 0, 1), th), 0, 1)
    return out


def _scale(img, tw, th):
    h, w = img.shape[:2]
    return np.asarray(Image.fromarray(_to_u8(img)).resize((tw or w, th or h), Image.LANCZOS)) / 255.0


def _draw_cut(img, xs, color=(255, 0, 0)):
    im = Image.fromarray(_to_u8(img))
    ImageDraw.Draw(im).line([(float(x), float(k)) for k, x in enumerate(xs)], fill=color, width=1)
    return np.asarray(im)


# ---------------------------------------------------------------- actions
def run_resize(image, wpct, hpct, xi1, xi2, methods, noise):
    if image is None:
        return [], "Upload an image first."
    img, note = _to_float(image)
    tw, th = _targets(img, wpct, hpct)
    if tw is None and th is None:
        return [(image, "original")], "Target is not smaller than the input (only shrinking is supported)."
    if noise > 0:
        img = img + np.random.default_rng(int(noise)).normal(0, 1e-9, img.shape)
    gallery, rows = [(_to_u8(img), f"Original {img.shape[1]}x{img.shape[0]}")], []
    p = Params(xi1=xi1, xi2=xi2)
    for m in methods:
        t = time.time()
        if m == "Calligraphic Cut":
            out = resize(img, tw, th, p)
        elif m == "Seam Carving":
            out = _sc_resize(img, tw, th)
        else:
            out = _scale(img, tw, th)
        dt = time.time() - t
        gallery.append((_to_u8(out), f"{m} {out.shape[1]}x{out.shape[0]}"))
        rows.append(f"| {m} | {out.shape[1]}x{out.shape[0]} | {dt:.2f} s |")
    md = (f"**{img.shape[1]}x{img.shape[0]} → {tw or img.shape[1]}x{th or img.shape[0]}**{note}  "
          f"ξ1={xi1} ξ2={xi2}" + (f"  · perturbation seed {int(noise)}" if noise > 0 else "") +
          "\n\n| method | size | time |\n|---|---|---|\n" + "\n".join(rows))
    return gallery, md


def run_progressive(image, wpct, frames, xi1, xi2, show_cut):
    if image is None:
        return [], "Upload an image first."
    img, note = _to_float(image)
    w = img.shape[1]
    tw = max(2, round(w * wpct / 100))
    if tw >= w:
        return [], "Target width must be smaller than the input."
    frames = int(frames)
    widths = sorted({round(w - (w - tw) * i / frames) for i in range(frames + 1)}, reverse=True)
    p = Params(xi1=xi1, xi2=xi2)
    cur, out, t = img, [], time.time()
    for fw in widths:
        if fw < cur.shape[1]:
            cur = resize(cur, fw, None, p)     # chunked == one run: the engine re-derives the same state
        shown = _to_u8(cur)
        if show_cut and cur.shape[1] > tw:
            shown = _draw_cut(cur, best_cut(cur, p)[1])
        out.append((shown, f"{cur.shape[1]} px ({100 * cur.shape[1] / w:.0f}%)"))
    return out, f"{len(out)} frames, {w}→{tw} px in {time.time() - t:.1f} s{note}. Red = next cut."


# ---------------------------------------------------------------- UI
def build():
    import gradio as gr

    examples = [str(EXAMPLES_DIR / f) for f in ("teaser_orig.jpg", "fig2_orig.png", "face_orig.png")
                if (EXAMPLES_DIR / f).exists()]
    with gr.Blocks(title="Calligraphic Cutting") as demo:
        gr.Markdown("# Calligraphic Cutting — local lab\n"
                    "Python re-implementation of Wang & Akleman, CGI 2013. Shrink-only. "
                    "ξ1: flat-region threshold on |∇F| (luma in [0,1]); ξ2: near-horizontal-edge threshold on |n_x|.")
        with gr.Tab("Resize & compare"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp = gr.Image(label="Input", type="numpy")
                    wp = gr.Slider(10, 100, 50, step=1, label="Width %")
                    hp = gr.Slider(10, 100, 100, step=1, label="Height %")
                    x1 = gr.Slider(0.0005, 0.03, 0.003, step=0.0005, label="ξ1")
                    x2 = gr.Slider(0.05, 0.6, 0.3, step=0.01, label="ξ2")
                    meth = gr.CheckboxGroup(["Calligraphic Cut", "Seam Carving", "Scaling"],
                                            value=["Calligraphic Cut", "Seam Carving", "Scaling"], label="Methods")
                    noise = gr.Number(0, precision=0, label="Perturbation seed (0 = off; adds 1e-9 noise)")
                    go = gr.Button("Run", variant="primary")
                    if examples:
                        gr.Examples(examples, inputs=inp)
                with gr.Column(scale=2):
                    gal = gr.Gallery(label="Results", columns=4, height=640, object_fit="contain", format="png")
                    info = gr.Markdown()
            go.click(run_resize, [inp, wp, hp, x1, x2, meth, noise], [gal, info])
        with gr.Tab("Progressive"):
            with gr.Row():
                with gr.Column(scale=1):
                    inp2 = gr.Image(label="Input", type="numpy")
                    wp2 = gr.Slider(10, 99, 50, step=1, label="Final width %")
                    fr = gr.Slider(2, 24, 8, step=1, label="Frames")
                    x12 = gr.Slider(0.0005, 0.03, 0.003, step=0.0005, label="ξ1")
                    x22 = gr.Slider(0.05, 0.6, 0.3, step=0.01, label="ξ2")
                    cut = gr.Checkbox(True, label="Overlay next cut (red)")
                    go2 = gr.Button("Run", variant="primary")
                    if examples:
                        gr.Examples(examples, inputs=inp2)
                with gr.Column(scale=2):
                    gal2 = gr.Gallery(label="Frames", columns=4, height=640, object_fit="contain", format="png")
                    info2 = gr.Markdown()
            go2.click(run_progressive, [inp2, wp2, fr, x12, x22, cut], [gal2, info2])
    return demo


def main():
    ap = argparse.ArgumentParser(description="Calligraphic Cutting local server")
    ap.add_argument("--host", default=os.environ.get("CC_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CC_PORT", 8097)))
    a = ap.parse_args()
    from .core import resize as _r
    _r(np.random.rand(16, 24, 3), 20)          # JIT warm-up (cached after first build)
    build().queue(default_concurrency_limit=2).launch(
        server_name=a.host, server_port=a.port, root_path=os.environ.get("CC_ROOT_PATH", ""),
        show_error=True)


if __name__ == "__main__":
    main()
