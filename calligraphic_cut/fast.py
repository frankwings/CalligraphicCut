"""numba kernels mirroring core.py (numpy reference) operation-for-operation."""
import numpy as np
from numba import njit


@njit(cache=True)
def build_axes(gx, gy, mag, xi1, xi2, cap):
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
                a = -gy[k, n] / gx[k, n]
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
