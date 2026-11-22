"""Calligraphic Cutting (Wang & Akleman, CGI 2013) -- numpy reference implementation.

Conventions (array coordinates):
  * image is H x W (x 3), row index k grows downward, column index n grows rightward.
  * Point rows k = 0..H are the horizontal lines between pixel rows; pixel row k
    lies between point rows k and k+1.  Every point row holds exactly W points,
    one per column (paper Sec. 4.2, step 3: merge + re-initialise).
  * A cut is a piecewise-linear medial axis xs[0..H]; inside pixel row k it is
    x(t) = (1-t) xs[k] + t xs[k+1].  Removing a width-1 strip centred on it shrinks
    the width by exactly one pixel (lambda = 1).
  * "Construction" runs in increasing k (paper: bottom-to-top), energy
    accumulation in decreasing k.  The opposite direction is obtained by running
    the same code on the vertically flipped image.

Decisions where the paper is silent / inconsistent (see docs/DECISIONS.md):
  D1 xi1, xi2 are free parameters (defaults tuned later).
  D2 Region-3 ("unreliable tangent") test uses |n_x| <= xi2 (paper prints n.(0,1)).
  D3 gradient = Sobel on luminance (option: sum of per-channel magnitudes).
  D4 alpha not clamped by default (|alpha| < 1/xi2 implicitly); optional cap.
  D5 chain local energy = max pixel energy along the row segment to the target.
  D6 full recomputation after every cut.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# ----------------------------------------------------------------------------
# gradients / energy
# ----------------------------------------------------------------------------
_LUMA = np.array([0.299, 0.587, 0.114])


def to_gray(img: np.ndarray) -> np.ndarray:
    return img @ _LUMA if img.ndim == 3 else img


def gradients(img: np.ndarray, mode: str = "luma"):
    """Return (gx, gy, mag) as derivatives per pixel (Sobel / 8)."""
    if mode == "luma":
        g = to_gray(img)
        gx = ndimage.sobel(g, axis=1, mode="nearest") / 8.0
        gy = ndimage.sobel(g, axis=0, mode="nearest") / 8.0
        return gx, gy, np.hypot(gx, gy)
    if mode == "rgb":
        # direction from the channel-summed structure (sign-aligned to luma),
        # magnitude = sum of per-channel magnitudes
        chans = [img[..., c] for c in range(img.shape[2])] if img.ndim == 3 else [img]
        gxs = [ndimage.sobel(c, axis=1, mode="nearest") / 8.0 for c in chans]
        gys = [ndimage.sobel(c, axis=0, mode="nearest") / 8.0 for c in chans]
        mag = sum(np.hypot(a, b) for a, b in zip(gxs, gys))
        gx, gy = sum(gxs), sum(gys)
        n = np.hypot(gx, gy) + 1e-12
        return gx / n * mag, gy / n * mag, mag
    raise ValueError(mode)


@dataclass
class Params:
    xi1: float = 0.003         # |grad| <= xi1  -> region 1 (flat), tangent unreliable
    xi2: float = 0.3           # |n_x|  <= xi2  -> region 3 (near-horizontal edge), unreliable
    grad_mode: str = "luma"
    alpha_cap: float | None = None   # optional |alpha| clamp (D4)
    link_energy: str = "pixel"       # 'pixel' (paper) or 'segmax' for reliable links too
    backend: str = "numba"           # 'numba' (fast) or 'numpy' (reference)


# ----------------------------------------------------------------------------
# medial-axis construction + energy accumulation (one direction)
# ----------------------------------------------------------------------------
def _runs(mask: np.ndarray):
    """(start, end_inclusive) of runs of True."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    return starts, ends


def build_axes(gx, gy, mag, p: Params):
    """Paper Sec 4.2.  Returns X (H+1, W) point x-positions and
    link (H, W): target column in the next row, -1 for chain members (resolved later)."""
    H, W = mag.shape
    X = np.empty((H + 1, W))
    link = np.empty((H, W), dtype=np.int64)
    cols = np.arange(W)
    x = cols + 0.5
    for k in range(H):
        X[k] = x
        m = mag[k]
        nx = np.abs(gx[k]) / np.maximum(m, 1e-12)
        rel = (m > p.xi1) & (nx > p.xi2)
        alpha = np.zeros(W)
        alpha[rel] = -gy[k, rel] / gx[k, rel]
        if p.alpha_cap is not None:
            np.clip(alpha, -p.alpha_cap, p.alpha_cap, out=alpha)
        xn = np.clip(x + alpha, 0.5, W - 0.5)              # step 4 (clamp)
        col = np.minimum(np.floor(xn).astype(np.int64), W - 1)
        cnt = np.bincount(col[rel], minlength=W)
        s = np.bincount(col[rel], weights=xn[rel], minlength=W)
        x = np.where(cnt > 0, s / np.maximum(cnt, 1), cols + 0.5)  # step 3 merge/init
        lk = np.where(rel, col, -1)
        # unreliable: isolated -> straight up ; chains (len>=2) -> resolved in energy pass
        st, en = _runs(~rel)
        iso = st[st == en]
        lk[iso] = iso
        link[k] = lk
    X[H] = x
    return X, link


def accumulate(link, e, p: Params, X=None):
    """Paper Sec 4.3.  Fills chain links in place; returns E0 (W,) total energy per start."""
    H, W = e.shape
    E = np.zeros(W)
    for k in range(H - 1, -1, -1):
        lk = link[k]
        ek = e[k]
        En = np.empty(W)
        ok = lk >= 0
        if p.link_energy == "pixel":
            En[ok] = E[lk[ok]] + ek[ok]
        else:  # max energy over the columns swept between n and target
            idx = np.flatnonzero(ok)
            for n in idx:
                a, b = sorted((n, lk[n]))
                En[n] = E[lk[n]] + ek[a:b + 1].max()
        st, en = _runs(~ok)
        for a, b in zip(st, en):
            m = a + int(np.argmin(E[a:b + 1]))
            seg = ek[a:b + 1]
            j = m - a
            loc = np.empty(b - a + 1)
            loc[j:] = np.maximum.accumulate(seg[j:])
            loc[:j + 1] = np.maximum.accumulate(seg[:j + 1][::-1])[::-1]
            En[a:b + 1] = E[m] + loc
            lk[a:b + 1] = m
        E = En
    return E


def trace(X, link, n0):
    H = link.shape[0]
    xs = np.empty(H + 1)
    c = n0
    xs[0] = X[0, c]
    for k in range(H):
        c = link[k, c]
        xs[k + 1] = X[k + 1, c]
    return xs


def best_cut_one_dir(img, p: Params):
    gx, gy, mag = gradients(img, p.grad_mode)
    if p.backend == "numba" and p.link_energy == "pixel":
        from . import fast
        X, link = fast.build_axes(gx, gy, mag, p.xi1, p.xi2, p.alpha_cap or 0.0)
        E = fast.accumulate(link, mag)
        n0 = int(np.argmin(E))
        return float(E[n0]), fast.trace(X, link, n0)
    X, link = build_axes(gx, gy, mag, p)
    E = accumulate(link, mag, p)
    n0 = int(np.argmin(E))
    return float(E[n0]), trace(X, link, n0)


def best_cut(img, p: Params):
    """Both construction directions; returns (energy, xs[0..H])."""
    e1, xs1 = best_cut_one_dir(img, p)
    e2, xs2 = best_cut_one_dir(img[::-1], p)
    if e2 < e1:
        return e2, xs2[::-1].copy()
    return e1, xs1


# ----------------------------------------------------------------------------
# removal with area-weighted colour interpolation (Sec 4.1)
# ----------------------------------------------------------------------------
def _G(z):
    """Antiderivative of clip(z, 0, 1)."""
    return np.where(z <= 0, 0.0, np.where(z >= 1, z - 0.5, 0.5 * z * z))


def left_weights(c0, c1, Wout):
    """a_j = int_0^1 clip(c(t) - j, 0, 1) dt with c linear from c0 to c1 (vectorised over j)."""
    j = np.arange(Wout)
    d = c1 - c0
    if abs(d) < 1e-9:
        return np.clip(c0 - j, 0.0, 1.0)
    return (_G(c1 - j) - _G(c0 - j)) / d


def remove_cut(img, xs, backend="numba"):
    """Remove the width-1 strip around medial axis xs; returns H x (W-1) image."""
    if backend == "numba":
        from . import fast
        im = np.ascontiguousarray(img, dtype=np.float64)
        if im.ndim == 2:
            return fast.remove_cut(im[..., None], xs)[..., 0]
        return fast.remove_cut(im, xs)
    H, W = img.shape[:2]
    out = np.empty((H, W - 1) + img.shape[2:], dtype=np.float64)
    for k in range(H):
        a = left_weights(xs[k] - 0.5, xs[k + 1] - 0.5, W - 1)
        if img.ndim == 3:
            a = a[:, None]
        out[k] = a * img[k, :-1] + (1 - a) * img[k, 1:]
    return out


def resize_width(img, target_w, p: Params | None = None, record=None):
    p = p or Params()
    cur = np.asarray(img, dtype=np.float64)
    while cur.shape[1] > target_w:
        _, xs = best_cut(cur, p)
        if record is not None:
            record.append(xs)
        cur = remove_cut(cur, xs, p.backend)
    return cur


def resize(img, target_w=None, target_h=None, p: Params | None = None):
    out = np.asarray(img, dtype=np.float64)
    if target_w is not None:
        out = resize_width(out, target_w, p)
    if target_h is not None:
        t = np.swapaxes(out, 0, 1)
        out = np.swapaxes(resize_width(t, target_h, p), 0, 1)
    return out
