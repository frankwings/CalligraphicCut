import numpy as np

from calligraphic_cut import Params, remove_cut, resize_width
from calligraphic_cut.core import left_weights
from calligraphic_cut import seam_carving as sc


def test_left_weights_vertical_integer_cut():
    a = left_weights(3.0, 3.0, 8)
    assert np.allclose(a, [1, 1, 1, 0, 0, 0, 0, 0])


def test_left_weights_slanted_is_area():
    # c goes 2 -> 4 within one row: pixel 2 is left of the line on a triangle of area 3/4? check numerically
    c0, c1 = 2.0, 4.0
    a = left_weights(c0, c1, 6)
    t = np.linspace(0, 1, 200001)
    ref = [np.trapezoid(np.clip(c0 + (c1 - c0) * t - j, 0, 1), t) for j in range(6)]
    assert np.allclose(a, ref, atol=1e-6)


def test_vertical_cut_equals_column_delete():
    rng = np.random.default_rng(0)
    img = rng.random((5, 7, 3))
    xs = np.full(6, 3.5)  # medial axis through centre of column 3
    out = remove_cut(img, xs)
    assert np.allclose(out, np.delete(img, 3, axis=1))


def test_constant_image_stays_constant():
    img = np.full((20, 30, 3), 0.4)
    out = resize_width(img, 20)
    assert out.shape == (20, 20, 3) and np.allclose(out, 0.4)


def _edge_fit(img):
    """Fit a line to the sub-pixel 0.5 crossing of each row that has one; return (slope, rms)."""
    g = img.mean(axis=2)
    ys, xs = [], []
    for k, row in enumerate(g):
        idx = np.flatnonzero((row[:-1] <= 0.5) & (row[1:] > 0.5))
        if len(idx) != 1:
            continue
        j = idx[0]
        xs.append(j + (0.5 - row[j]) / (row[j + 1] - row[j]) + 0.5)
        ys.append(k)
    ys, xs = np.array(ys), np.array(xs)
    p = np.polyfit(ys, xs, 1)
    return p[0], float(np.sqrt(np.mean((np.polyval(p, ys) - xs) ** 2))), len(ys)


def _edge_image():
    H, W = 100, 80
    yy, xx = np.mgrid[0:H, 0:W] + 0.5
    img = np.clip((xx - (2 + 0.76 * yy)) + 0.5, 0, 1)  # edge spans the full width
    return np.repeat(img[..., None], 3, axis=2)


def test_type2_cut_preserves_straight_edge_slope():
    """Core claim: CC keeps a full-width straight edge straight with the same slope;
    seam carving (no low-energy path exists) bends/jags it."""
    img = _edge_image()
    s0, _, _ = _edge_fit(img)
    s_cc, r_cc, n_cc = _edge_fit(resize_width(img, 55, Params()))
    s_sc, r_sc, _ = _edge_fit(sc.resize_width(img, 55))
    assert n_cc > 50
    assert abs(s_cc - s0) < 0.02 and r_cc < 0.2, (s0, s_cc, r_cc)
    assert abs(s_sc - s0) > 0.1 or r_sc > 1.0, (s_sc, r_sc)  # baseline really fails
