# Implementation decisions (paper is silent / inconsistent)

Approved by the paper author, 2026-11-22.

| # | Gap in paper | Decision |
|---|---|---|
| D1 | xi1, xi2 have no values | free params; sweep1 (7 RetargetMe imgs, SSIM vs 2013 MatLab outputs) -> **xi1=0.003, xi2=0.3** (luma, Sobel/8, img in [0,1]). xi1=0.02 lets cuts slide through faces (skin |grad| ~0.004-0.009). |
| D2 | Region-3 test printed as n.(0,1) <= xi2 | implemented as \|n_x\| <= xi2 (near-horizontal edge => tangent unreliable) |
| D3 | gradient operator / colour | Sobel on luminance; 'rgb' option scored worse in sweep1 |
| D4 | alpha unbounded | not clamped (implicitly \|alpha\| < 1/xi2); optional alpha_cap |
| D5 | chain local energy "max along path" | max pixel energy over the row segment between chain point and its target |
| D6 | recompute policy | full recompute after every cut |

Metric note: SSIM/PSNR vs the 2013 outputs is alignment-sensitive (mean SSIM: best 0.47, SC 0.38);
use it for ranking only, final judgement is visual (out/sweep1_visual.png).

## Hold-out (cc_holdout1, 10 unseen RetargetMe images, 2026-11-22)
- xi2=0.3 is the robust choice: all four xi2=0.3 configs top the table (mean SSIM 0.414-0.435 vs SC 0.350).
- xi1 is flat within 0.001-0.012; keep 0.003 (beats SC on 8/10).
- Visual (out/holdout1_visual.png): python CC ~= 2013 MatLab on pigeons/glasses/girls/waterfall; SC breaks the man's head (pigeons) and glasses frame.

## D7 Sensitivity to tiny perturbations — accepted, no tie-breaking (author decision, 2026-11-22)
- Floating-point order (e.g. BLAS vs explicit luma, ~1e-16 on 20 % of pixels) changes single
  trajectories (teaser diverges at cut #101) but NOT the quality distribution
  (10-run medians identical: face 0.0306/0.0307, teaser 0.0651/0.0651) -> irrelevant.
- Input noise (1e-9) changes which cut wins near-ties (argmin ties, floor at column borders,
  xi1/xi2 thresholds). Inherent to greedy carving; every branch is artefact-free. Left as is.
- Consequence for evaluation: judge on medians over perturbed runs (tests/test_e2e_paper.py).
