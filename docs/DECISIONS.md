# Implementation decisions (paper is silent / inconsistent)

Boss approved 2026-11-22.

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
