"""End-to-end tests that pin the implementation to the paper (CGI 2013).

Two layers:
  A. Paper alignment -- full resize on the paper's own figures, asserting the claims the paper
     makes, measured against the 2013 MatLab outputs (tests/data/paper/*_cc2013*).  Every claim is
     also checked to FAIL for seam carving, so the test really discriminates.
     Greedy cut selection is chaotic: a 1e-16 change in luminance flips near-tied argmins and the
     whole cut sequence diverges (measured 2026-11-22: face blur-L1 0.029..0.041 over 1e-9 input
     noise).  So claims are judged on the MEDIAN of N_RUNS runs (run 0 = exact input, runs 1.. =
     input + N(0, 1e-9) with fixed seeds), never on a single trajectory.
  B. Golden regression -- the exact uint8 outputs of the current implementation
     (tests/data/golden/).  Any behavioural change shows up here.  If the change is intended:
         UPDATE_GOLDEN=1 python3 -m pytest tests/test_e2e_paper.py
     then re-check layer A and the visual comparison before committing the new goldens.
"""
import os

import numpy as np
import pytest
from scipy import ndimage

from calligraphic_cut import Params, resize_width
from calligraphic_cut import seam_carving as sc
from calligraphic_cut.io import load, save

D = os.path.join(os.path.dirname(__file__), "data")
P = os.path.join(D, "paper")
G = os.path.join(D, "golden")

# (key, original, 2013 MatLab result, target width)
CASES = {
    "fig2": ("fig2_orig.png", "fig2_cc2013.png", 189),          # paper Fig. 2, 512 -> 189
    "teaser": ("teaser_orig.jpg", "teaser_cc2013_50.png", 257),  # paper Fig. 1, 50 %
    "face": ("face_orig.png", "face_cc2013_75.png", 294),        # paper Fig. 12, 75 %
}


N_RUNS = 7


@pytest.fixture(scope="module")
def runs():
    out = {}
    for k, (o, r, tw) in CASES.items():
        orig = load(os.path.join(P, o))
        ccs = [resize_width(orig if i == 0 else orig + np.random.default_rng(i).normal(0, 1e-9, orig.shape),
                            tw, Params()) for i in range(N_RUNS)]
        out[k] = dict(orig=orig, ref=load(os.path.join(P, r)), cc=ccs[0], ccs=ccs,
                      sc=sc.resize_width(orig, tw))
    return out


# ---------------------------------------------------------------- A. paper alignment
BLUE = np.array([105, 139, 159]) / 255.0
YELLOW = np.array([245, 238, 49]) / 255.0


def _blue(im):
    return np.linalg.norm(im - BLUE, axis=2) < np.linalg.norm(im - YELLOW, axis=2)


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_fig2_cc_removes_only_low_energy_region(runs):
    """Fig. 2: calligraphic cuts reach every low-energy (yellow) region, so all shapes survive intact."""
    r = runs["fig2"]
    for cc in r["ccs"]:                                              # must hold for EVERY run
        assert _blue(cc).sum() == _blue(r["orig"]).sum()             # not one shape pixel lost
        assert ndimage.label(_blue(cc))[1] == 5                      # 5 ellipses, none split
        assert _iou(_blue(cc), _blue(r["ref"])) > 0.99                # same as 2013 result
    assert _iou(_blue(r["sc"]), _blue(r["ref"])) < 0.8               # SC cannot do it


def _blur_l1(a, b, s):
    return np.abs(ndimage.gaussian_filter(a.mean(2), s) - ndimage.gaussian_filter(b.mean(2), s)).mean()


def test_teaser_matches_2013_better_than_seam_carving(runs):
    """Fig. 1 (extreme 50 % resize of a facade): result is structurally closer to the 2013 CC output than SC is."""
    r = runs["teaser"]
    cc = np.median([_blur_l1(c, r["ref"], 4) for c in r["ccs"]])
    s = _blur_l1(r["sc"], r["ref"], 4)
    assert cc < 0.95 * s, (cc, s)


def test_face_preserved_like_2013(runs):
    """Fig. 12: face survives 75 % resize; SC cuts through it."""
    r = runs["face"]
    cc = np.median([_blur_l1(c, r["ref"], 2) for c in r["ccs"]])
    s = _blur_l1(r["sc"], r["ref"], 2)
    assert cc < 0.04 and cc < 0.6 * s, (cc, s)


# ---------------------------------------------------------------- B. golden regression
@pytest.mark.parametrize("key", list(CASES))
def test_golden_regression(runs, key):
    path = os.path.join(G, f"{key}_cc.png")
    got = np.clip(np.rint(runs[key]["cc"] * 255), 0, 255).astype(np.uint8)
    if os.environ.get("UPDATE_GOLDEN") == "1" or not os.path.exists(path):
        os.makedirs(G, exist_ok=True)
        save(path, runs[key]["cc"])
        pytest.skip(f"golden written: {path}")
    want = np.rint(load(path) * 255).astype(np.uint8)
    assert got.shape == want.shape
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.max() == 0, f"{key}: {np.count_nonzero(diff)} pixels changed (max {diff.max()})"
