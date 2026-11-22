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
