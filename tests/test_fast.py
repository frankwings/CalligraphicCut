import numpy as np

from calligraphic_cut import Params
from calligraphic_cut.core import best_cut, remove_cut, resize_width


def _img(seed=0, H=40, W=60):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    base = 0.5 + 0.4 * np.sin(xx / 7.0 + yy / 11.0) * (xx > 20) + 0.05 * rng.random((H, W))
    flat = np.where((yy > 10) & (yy < 18), 0.3, base)          # flat band -> horizontal chains
    return np.stack([flat, flat ** 2, 1 - flat], axis=2)


def test_numba_matches_numpy_cut_by_cut():
    for seed in range(3):
        img = _img(seed)
        for p_np, p_nb in [(Params(backend="numpy"), Params(backend="numba")),
                           (Params(backend="numpy", xi1=0.02, xi2=0.15), Params(backend="numba", xi1=0.02, xi2=0.15))]:
            a = b = img
            for _ in range(15):
                ea, xa = best_cut(a, p_np)
                eb, xb = best_cut(b, p_nb)
                assert np.array_equal(xa, xb) and ea == eb
                a, b = remove_cut(a, xa, "numpy"), remove_cut(b, xb, "numba")
                assert np.array_equal(a, b)


def test_numba_full_resize_equal():
    img = _img(5)
    assert np.array_equal(resize_width(img, 35, Params(backend="numpy")), resize_width(img, 35, Params(backend="numba")))


def test_incremental_gradients_equal_full_recompute():
    """After every cut the band-updated luma/gradients must equal a full recompute, bit for bit."""
    from calligraphic_cut import fast
    from calligraphic_cut.core import gradients, to_gray
    img = _img(7, 50, 70)
    L, gx, gy, mag = fast.init_state(img)
    cur = img
    for _ in range(25):
        _, xs = fast.best_cut_state(gx, gy, mag, 0.003, 0.3, 0.0)
        cur, L, gx, gy, mag = fast.cut_update(cur, L, gx, gy, mag, xs)
        rx, ry, rm = gradients(cur)
        assert np.array_equal(L, to_gray(cur))
        assert np.array_equal(gx, rx) and np.array_equal(gy, ry) and np.array_equal(mag, rm)


def test_fast_resize_equals_reference_on_paper_images():
    import os
    from calligraphic_cut.io import load
    d = os.path.join(os.path.dirname(__file__), "data", "paper")
    for f, tw in [("face_orig.png", 330), ("fig2_orig.png", 400)]:
        img = load(os.path.join(d, f))
        assert np.array_equal(resize_width(img, tw, Params(backend="numpy")),
                              resize_width(img, tw, Params(backend="numba")))
