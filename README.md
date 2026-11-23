# CalligraphicCut

Python re-implementation of **Calligraphic Cutting: Discrete Image Resizing with Cuts in Continuous Domain**
(Youyou Wang, Ergun Akleman — CGI 2013). The original MatLab code was lost; this version was rebuilt
from the paper, its LaTeX sources, figures and the 2013 result images (2026-11).

## Idea in one paragraph
Seam carving removes 8-connected pixel paths, so once flat regions are used up it must cut across
edges (jaggies, broken lines). Calligraphic cuts are defined in the continuous domain: a width-λ strip
around a medial axis `x = f(y)` with *no* slope limit. The medial axis follows the image isocurves
(tangent ⟂ ∇F — "type-2" cuts keep slopes/G1), passes vertically through near-horizontal edges /
extrema ("type-3"), and may run horizontally through flat areas ("type-1"/horizontal chains). Pixels are
piecewise-constant squares, the axis is piecewise linear, so after removal each new pixel is an
area-weighted mix of exactly two old pixels.

## Usage
```bash
pip install numpy scipy numba pillow scikit-image   # scikit-image only for scripts/sweep & run_all
python3 scripts/cc_resize.py in.png out.png --width 300          # shrink width
python3 scripts/cc_resize.py in.png out.png --height 400         # shrink height (transpose)
python3 -m pytest -q
```
```python
from calligraphic_cut import Params, resize
out = resize(img_float_rgb_0to1, target_w=300, p=Params(xi1=0.003, xi2=0.3))
```

## Layout
| path | content |
|---|---|
| `calligraphic_cut/core.py` | numpy reference: gradients, medial-axis construction (Sec 4.2), energy accumulation + horizontal chains (4.3), both directions, area-weighted removal (4.1) |
| `calligraphic_cut/fast.py` | numba kernels, **bit-identical** to the reference (tested), 10–16× faster |
| `calligraphic_cut/seam_carving.py` | baseline: original seam carving (Avidan & Shamir 2007) |
| `scripts/` | `cc_resize` CLI, `teaser`, `fig2`, `sweep` (param search), `run_all` (RetargetMe batch), `make_montages`, `viz_cuts` |
| `docs/DECISIONS.md` | every place the paper is silent/inconsistent and what we chose |

## Validation (vs. the 2013 MatLab outputs)
* Paper Fig. 2 (two-colour ellipses, 512→189): identical behaviour — ellipses intact, SC breaks them.
* Teaser (Toledo cathedral, 50% / 66%): pointed arch moves down intact, doors preserved, as in the paper.
* 36 RetargetMe cases (incl. one vertical): CC-python closer to the 2013 result than SC in 28/36
  (mean SSIM 0.398 vs 0.328). SSIM vs a retargeted reference is alignment-sensitive → used for ranking only.
* Speed (numba, 1 core): 512×683 → 50% in 3.8 s; 1024-wide → 50% ≈ 45–55 s.

Defaults `xi1=0.003, xi2=0.3` (luminance Sobel/8, image in [0,1]); tuned on 7 images, confirmed on 10 hold-out.

## Tests
* `tests/test_core.py`, `tests/test_fast.py` — unit tests; numba == numpy bit-exact, cut by cut.
* `tests/test_e2e_paper.py` — **end-to-end, pinned to the paper** (~15 s):
  * A. paper claims on the paper's own figures vs the 2013 MatLab outputs (`tests/data/paper/`):
    Fig. 2 every shape pixel kept + mask IoU > 0.99 with 2013 (SC < 0.8); teaser 50 % structurally closer
    to 2013 than SC; face 75 % preserved (Fig. 12). Each claim is also asserted to fail for SC.
  * B. golden regression: exact uint8 outputs in `tests/data/golden/`. Intended change →
    `UPDATE_GOLDEN=1 python3 -m pytest tests/test_e2e_paper.py`, re-check A + visuals, commit goldens.
  * Mutation-checked: xi1=0.02 or tangent-following disabled (alpha=0) both make the suite fail.

## Not (yet) covered
Enlarging (paper does not define it); improved seam carving (forward energy) baseline; saliency maps.
