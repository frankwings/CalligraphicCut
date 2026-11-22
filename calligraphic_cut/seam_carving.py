"""Baseline: original seam carving (Avidan & Shamir 2007), backward e1 energy."""
import numpy as np

from .core import to_gray


def e1(img):
    g = to_gray(img)
    return np.abs(np.gradient(g, axis=1)) + np.abs(np.gradient(g, axis=0))


def find_seam(e):
    H, W = e.shape
    M = e.copy()
    back = np.zeros((H, W), dtype=np.int64)
    for k in range(1, H):
        prev = np.pad(M[k - 1], 1, constant_values=np.inf)
        stack = np.stack([prev[:-2], prev[1:-1], prev[2:]])
        a = np.argmin(stack, axis=0)
        back[k] = np.arange(W) + a - 1
        M[k] += stack[a, np.arange(W)]
    s = np.empty(H, dtype=np.int64)
    s[-1] = int(np.argmin(M[-1]))
    for k in range(H - 1, 0, -1):
        s[k - 1] = back[k, s[k]]
    return s


def remove_seam(img, s):
    H, W = img.shape[:2]
    mask = np.ones((H, W), bool)
    mask[np.arange(H), s] = False
    return img[mask].reshape((H, W - 1) + img.shape[2:])


def resize_width(img, target_w):
    cur = np.asarray(img, dtype=np.float64)
    while cur.shape[1] > target_w:
        cur = remove_seam(cur, find_seam(e1(cur)))
    return cur
