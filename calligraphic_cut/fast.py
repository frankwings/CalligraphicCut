"""numba kernels mirroring core.py (numpy reference) operation-for-operation.

Bit-exactness with the reference is a hard requirement (tests/test_fast.py):
  * same floating-point expression order everywhere, no fastmath;
  * `resize_width` below keeps luma + gradients as state and after each cut recomputes them only
    in a narrow band around the cut -- outside the band the 3x3 Sobel neighbourhood is an exact
    (possibly 1-column-shifted) copy of the old one, so copying the old value is exact;
  * the opposite cut direction uses the flipped views (gx[::-1], -gy[::-1], mag[::-1]), exact
    because core.sobel is order-symmetric.
"""
import math

import numpy as np
from numba import njit


@njit(cache=True)
def build_axes(gx, gy, mag, xi1, xi2, cap, sgn=1.0):
    """sgn=-1 with row-flipped views == running on the vertically flipped image."""
    H, W = mag.shape
    X = np.empty((H + 1, W))
    link = np.empty((H, W), dtype=np.int64)
    x = np.empty(W)
    for n in range(W):
        x[n] = n + 0.5
    rel = np.empty(W, dtype=np.bool_)
    cnt = np.empty(W, dtype=np.int64)
    s = np.empty(W)
    for k in range(H):
        for n in range(W):
            X[k, n] = x[n]
            cnt[n] = 0
            s[n] = 0.0
        for n in range(W):
            m = mag[k, n]
            nx = abs(gx[k, n]) / max(m, 1e-12)
            r = (m > xi1) and (nx > xi2)
            rel[n] = r
            if r:
                a = -(sgn * gy[k, n]) / gx[k, n]
                if cap > 0:
                    a = min(max(a, -cap), cap)
                xn = min(max(x[n] + a, 0.5), W - 0.5)
                c = min(int(np.floor(xn)), W - 1)
                cnt[c] += 1
                s[c] += xn
                link[k, n] = c
            else:
                link[k, n] = -1
        for n in range(W):
            if not rel[n]:
                left_ok = n == 0 or rel[n - 1]
                right_ok = n == W - 1 or rel[n + 1]
                if left_ok and right_ok:
                    link[k, n] = n
        for n in range(W):
            x[n] = s[n] / cnt[n] if cnt[n] > 0 else n + 0.5
    for n in range(W):
        X[H, n] = x[n]
    return X, link


@njit(cache=True)
def accumulate(link, e):
    H, W = e.shape
    E = np.zeros(W)
    En = np.empty(W)
    for k in range(H - 1, -1, -1):
        n = 0
        while n < W:
            t = link[k, n]
            if t >= 0:
                En[n] = E[t] + e[k, n]
                n += 1
                continue
            a = n
            while n < W and link[k, n] < 0:
                n += 1
            b = n - 1
            m = a
            for j in range(a + 1, b + 1):
                if E[j] < E[m]:
                    m = j
            run = 0.0
            for j in range(m, b + 1):
                run = e[k, j] if j == m else max(run, e[k, j])
                En[j] = E[m] + run
                link[k, j] = m
            run = 0.0
            for j in range(m, a - 1, -1):
                run = e[k, j] if j == m else max(run, e[k, j])
                En[j] = E[m] + run
                link[k, j] = m
        for j in range(W):
            E[j] = En[j]
    return E


@njit(cache=True)
def trace(X, link, n0):
    H = link.shape[0]
    xs = np.empty(H + 1)
    c = n0
    xs[0] = X[0, c]
    for k in range(H):
        c = link[k, c]
        xs[k + 1] = X[k + 1, c]
    return xs


@njit(cache=True)
def _G(z):
    if z <= 0:
        return 0.0
    if z >= 1:
        return z - 0.5
    return 0.5 * z * z


@njit(cache=True)
def remove_cut(img, xs):
    H, W, C = img.shape
    out = np.empty((H, W - 1, C))
    for k in range(H):
        c0 = xs[k] - 0.5
        c1 = xs[k + 1] - 0.5
        d = c1 - c0
        for j in range(W - 1):
            if abs(d) < 1e-9:
                a = min(max(c0 - j, 0.0), 1.0)
            else:
                a = (_G(c1 - j) - _G(c0 - j)) / d
            for c in range(C):
                out[k, j, c] = a * img[k, j, c] + (1 - a) * img[k, j + 1, c]
    return out


# ----------------------------------------------------------------------------
# stateful resize loop with incremental gradient update
# ----------------------------------------------------------------------------
@njit(cache=True, inline="always")
def _luma(r, g, b):
    return (0.299 * r + 0.587 * g) + 0.114 * b


@njit(cache=True, inline="always")
def _grad_at(L, k, n, H, W):
    ku = max(k - 1, 0)
    kd = min(k + 1, H - 1)
    nl = max(n - 1, 0)
    nr = min(n + 1, W - 1)
    gx = ((L[ku, nr] - L[ku, nl]) + (L[kd, nr] - L[kd, nl]) + 2.0 * (L[k, nr] - L[k, nl])) / 8.0
    gy = ((L[kd, nl] - L[ku, nl]) + (L[kd, nr] - L[ku, nr]) + 2.0 * (L[kd, n] - L[ku, n])) / 8.0
    return gx, gy


@njit(cache=True)
def init_state(img):
    H, W, C = img.shape
    L = np.empty((H, W))
    for k in range(H):
        for n in range(W):
            L[k, n] = _luma(img[k, n, 0], img[k, n, 1], img[k, n, 2])
    gx = np.empty((H, W))
    gy = np.empty((H, W))
    mag = np.empty((H, W))
    for k in range(H):
        for n in range(W):
            a, b = _grad_at(L, k, n, H, W)
            gx[k, n] = a
            gy[k, n] = b
            mag[k, n] = math.hypot(a, b)
    return L, gx, gy, mag


@njit(cache=True)
def cut_update(img, L, gx, gy, mag, xs):
    """remove_cut + exact incremental update of luma and gradients."""
    H, W, C = img.shape
    Wn = W - 1
    out = np.empty((H, Wn, C))
    Ln = np.empty((H, Wn))
    jl = np.empty(H, dtype=np.int64)   # first column with a < 1 (mixed or shifted)
    jr = np.empty(H, dtype=np.int64)   # last column with a > 0
    for k in range(H):
        c0 = xs[k] - 0.5
        c1 = xs[k + 1] - 0.5
        d = c1 - c0
        lo = Wn
        hi = -1
        for j in range(Wn):
            if abs(d) < 1e-9:
                a = min(max(c0 - j, 0.0), 1.0)
            else:
                a = (_G(c1 - j) - _G(c0 - j)) / d
            for c in range(C):
                out[k, j, c] = a * img[k, j, c] + (1 - a) * img[k, j + 1, c]
            if a < 1.0 and j < lo:
                lo = j
            if a > 0.0:
                hi = j
            if a == 1.0:
                Ln[k, j] = L[k, j]
            elif a == 0.0:
                Ln[k, j] = L[k, j + 1]
            else:
                Ln[k, j] = _luma(out[k, j, 0], out[k, j, 1], out[k, j, 2])
        jl[k] = lo
        jr[k] = hi
    gxn = np.empty((H, Wn))
    gyn = np.empty((H, Wn))
    magn = np.empty((H, Wn))
    for k in range(H):
        b0 = jl[k]
        b1 = jr[k]
        if k > 0:
            b0 = min(b0, jl[k - 1])
            b1 = max(b1, jr[k - 1])
        if k < H - 1:
            b0 = min(b0, jl[k + 1])
            b1 = max(b1, jr[k + 1])
        b0 = max(b0 - 1, 0)
        b1 = min(b1 + 1, Wn - 1)
        for j in range(Wn):
            if j < b0:
                gxn[k, j] = gx[k, j]
                gyn[k, j] = gy[k, j]
                magn[k, j] = mag[k, j]
            elif j > b1:
                gxn[k, j] = gx[k, j + 1]
                gyn[k, j] = gy[k, j + 1]
                magn[k, j] = mag[k, j + 1]
            else:
                a, b = _grad_at(Ln, k, j, H, Wn)
                gxn[k, j] = a
                gyn[k, j] = b
                magn[k, j] = math.hypot(a, b)
    return out, Ln, gxn, gyn, magn


@njit(cache=True)
def best_cut_state(gx, gy, mag, xi1, xi2, cap):
    X1, L1 = build_axes(gx, gy, mag, xi1, xi2, cap, 1.0)
    E1 = accumulate(L1, mag)
    n1 = np.argmin(E1)
    X2, L2 = build_axes(gx[::-1], gy[::-1], mag[::-1], xi1, xi2, cap, -1.0)
    E2 = accumulate(L2, mag[::-1])
    n2 = np.argmin(E2)
    if E2[n2] < E1[n1]:
        return E2[n2], trace(X2, L2, n2)[::-1].copy()
    return E1[n1], trace(X1, L1, n1)


@njit(cache=True)
def resize_width(img, target_w, xi1, xi2, cap):
    L, gx, gy, mag = init_state(img)
    cur = img
    while cur.shape[1] > target_w:
        _, xs = best_cut_state(gx, gy, mag, xi1, xi2, cap)
        cur, L, gx, gy, mag = cut_update(cur, L, gx, gy, mag, xs)
    return cur
