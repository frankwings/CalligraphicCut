# CalligraphicCut — Algorithm, Implementation & Architecture

Python re-implementation of *Calligraphic Cutting: Discrete Image Resizing with Cuts in Continuous
Domain* (Youyou Wang, Ergun Akleman — CGI 2013). Original MatLab code lost; rebuilt in 2026-11 from
the paper, all LaTeX versions (SIGGRAPH'12 → EG'13 → CGI'13), the EG'13 reviews/rebuttal, the
algorithm figures and the 2013 result images.

Companion docs: `DECISIONS.md` (every gap in the paper and what we chose), `../README.md` (usage).

---

## 1. Algorithm core

### 1.1 Model
* The image is a function `F : [0,X_M]×[0,Y_M] → [0,1]`; a discrete image is piecewise constant
  (each pixel a unit square).
* A **top-to-bottom calligraphic cut** is the strip
  `CC(f,λ) = {(x,y) | f(y) − λ/2 ≤ x < f(y) + λ/2}`, `f` any continuous function. Removing it and
  joining the two halves shrinks the width by λ. `(f(t),t)` is the **medial axis**.
* A **seam** is the special case `|df/dt| < 1` (8-connected path). Calligraphic cuts have no slope
  limit, so they can run (almost) horizontally — follow contours and reach every low-energy region.

### 1.2 Why it preserves geometry (paper §3.2)
| type | cut … | effect |
|---|---|---|
| 1 | never touches a boundary | nothing changes (best) |
| 2 | medial axis follows a boundary (tangent ⟂ ∇F) | boundary shifts by λ/2; slope, C1/G1 kept; curvature centre moves (circle → "football") |
| 3 | crosses a boundary at a critical point (extremum / horizontal edge) vertically | G0 kept, sharp points stay sharp, may create a corner |
| 4 | crosses a boundary elsewhere | G0 break: vertical (seam-carving) or sliding (zig-zag) discontinuity — avoid |

Preference: 1 > 2 > 3, never 4.

### 1.3 Regions → composite cut (paper §3.3)
With `n = ∇F/|∇F|`:
* **Region 1** `|∇F| ≤ ξ1` (flat): axis may be anything → "unreliable tangent", free routing.
* **Region 2** otherwise and `|n_x| > ξ2`: axis follows the isocurve, `(α,1)·∇F = 0` ⇒ `α = −F_y/F_x`.
* **Region 3** `|n_x| ≤ ξ2` (near-horizontal edge, extremum): axis goes straight up.
  (Paper prints `n·(0,1) ≤ ξ2`; we use `|n_x|` — see DECISIONS D2.)

### 1.4 Cost
`E1 = ∫_CC |∇F|` — the continuous version of original seam-carving energy. No saliency, by design.

---

## 2. Implementation details (discrete algorithm, paper §4)

Coordinates: array row `k` (down), column `n` (right). **Point rows** `k = 0..H` lie between pixel
rows; pixel row `k` is between point rows `k` and `k+1`. Each point row has exactly **W points, one per
column** (invariant maintained by merge/init).

### 2.1 Gradients (`core.sobel`, `fast._grad_at`)
* luma `L = (0.299R + 0.587G) + 0.114B` (explicit expression, fixed order).
* Sobel/8 with edge replication, written order-symmetric:
  `gx = ((dx_up + dx_down) + 2·dx_mid)/8`, `gy = ((dy_left + dy_right) + 2·dy_mid)/8`.
  Consequence (exact in IEEE): `sobel(L[::-1]) = (gx[::-1], −gy[::-1])`.
* energy `e = |∇F| = hypot(gx, gy)`; the same field drives both the tangent and the cost.

### 2.2 Medial-axis construction, one direction (`build_axes`, paper §4.2 steps 1–5)
```
x ← [0.5, 1.5, …, W−0.5]                                    # step 1
for k = 0..H−1:
    X[k] ← x
    for each point n:  reliable ⇔ |∇F|>ξ1 ∧ |n_x|>ξ2  (gradient of pixel (k,n))
        reliable:  x' = clip(x[n] + α, 0.5, W−0.5),  α = −g_y/g_x       # step 2 + step 4
                   col = min(floor(x'), W−1);  link[k,n] = col
    x_next[c] = mean of x' landing in column c, else c+0.5               # step 3 merge / init
    unreliable & isolated  → link[k,n] = n (straight up)
    unreliable runs (len≥2) = horizontal chains → link = −1, resolved in 2.3
X[H] ← x
```
All links point to the next point row, so the cuts form a **forest** (cuts can share upper parts).

### 2.3 Energy accumulation, opposite direction (`accumulate`, paper §4.3)
```
E ← 0 (row H, no pixel)
for k = H−1 … 0:
    linked n:          E'[n] = E[link[k,n]] + e[k,n]
    chain a..b:        m = argmin_{a≤j≤b} E[j]    (first minimum)
                       every chain point links to m;
                       E'[j] = E[m] + max(e[k, j..m])     # "max energy along the path" (D5)
    E ← E'
```
`E[n]` = total energy of the cut starting at column n. `trace` follows links to get `xs[0..H]`.

### 2.4 Both directions (`best_cut`)
Construction direction matters (paper reviewer Q1). Run 2.2–2.3 on the image and on its vertical
flip; take the lower minimum (ties → first direction). The flip reuses gradients (§2.1 identity) by
passing `gx[::-1], gy[::-1], mag[::-1]` and `sgn = −1` (α uses `sgn·g_y`).

### 2.5 Removal + colour interpolation (`remove_cut`, paper §4.1)
Inside pixel row k the axis is linear; left strip edge `c(t) = (1−t)(xs[k]−½) + t(xs[k+1]−½)`.
New pixel `j` = `a_j·P[j] + (1−a_j)·P[j+1]`,
`a_j = ∫₀¹ clip(c(t) − j, 0, 1) dt = (G(c1−j) − G(c0−j))/(c1−c0)`,
`G` = antiderivative of `clip(z,0,1)` (`0`, `z²/2`, `z−½`). Always exactly two source pixels; a
trapezoid/triangle area weight. `a_j` is monotone non-increasing in `j`.

### 2.6 Loop
λ = 1; one cut per iteration, everything recomputed (D6) until the target width. Height = transpose.

### 2.7 Parameters
`xi1 = 0.003`, `xi2 = 0.3` (luma in [0,1], Sobel/8). Tuned on 7 RetargetMe images (SSIM vs the 2013
outputs), confirmed on 10 hold-out. `xi1 = 0.02` lets cuts slide through faces.

---

## 3. Architecture

```
calligraphic_cut/
  core.py          numpy REFERENCE implementation + public API
                     to_gray, sobel, gradients, Params,
                     build_axes, accumulate, trace, best_cut_one_dir, best_cut,
                     left_weights, remove_cut, resize_width, resize
  fast.py          numba backend
                     kernels: build_axes(sgn), accumulate, trace, remove_cut
                     engine : init_state, cut_update (incremental), best_cut_state, resize_width
  seam_carving.py  baseline (Avidan & Shamir 2007, backward e1 energy)
  io.py            load/save float RGB in [0,1]
scripts/           cc_resize (CLI) · teaser · fig2 · sweep · run_all · make_montages · viz_cuts · compare
tests/             test_core · test_fast · test_e2e_paper ; data/paper (2013 refs) · data/golden
docs/              ARCHITECTURE.md (this) · DECISIONS.md
```

### 3.1 Data flow
```
img ─► luma L ─► sobel ─► (gx, gy, mag)
                              │
              ┌───────────────┴───────────────┐
        build_axes(+1)                 build_axes(views[::-1], −1)
        accumulate                     accumulate
              └──────► argmin both ◄───────────┘
                              │ xs[0..H]
                       remove_cut / cut_update ─► next iteration
```

### 3.2 Backends and dispatch
`Params.backend = "numba"` (default) or `"numpy"`.
* `resize_width` → `fast.resize_width` (whole loop in one njit call) when backend=numba,
  grad_mode=luma, link_energy=pixel, colour image, no cut recording.
* `best_cut` / `remove_cut` (public, used by viz) → numba kernels per call.
* Anything else (grad_mode=rgb, link_energy=segmax, recording) → numpy reference.

### 3.3 Exactness contract (numba ≡ numpy, bit for bit)
Required so the fast engine can be trusted and goldens are meaningful.
* Identical expression order everywhere; no `fastmath`; explicit luma (not BLAS — a BLAS dot gave
  1-ulp differences on 20 % of pixels).
* Merge averaging accumulates in increasing column order (= `np.bincount`).
* `argmin` → first minimum, as numpy.
* Tested cut-by-cut (`test_fast.py`), plus incremental state vs full recompute after every cut.

### 3.4 Incremental gradient update (`fast.cut_update`) — why it is exact
For each row `a_j` is 1 left of the cut, 0 right of it, mixed in between. Let `[jl_k, jr_k]` be the
mixed/switch columns of row k. Output pixel with `a=1` is bit-identical to `P[j]` (`1·x + 0·y = x`),
with `a=0` to `P[j+1]`; same for its luma. Sobel at (k,j) reads a 3×3 neighbourhood, so outside the
band `[min(jl_{k−1..k+1}) − 1, max(jr_{k−1..k+1}) + 1]` the neighbourhood is an unchanged copy
(left) or a 1-column-shifted copy (right) ⇒ copy the old gradient. Only the band is recomputed.
Border clamping is consistent on both sides.

### 3.5 Performance (single core)
| image | numpy ref | numba v1 (kernels) | numba v2 (engine) |
|---|---|---|---|
| face 392→294 | 7.6 s | 0.5 s | **0.16 s** |
| teaser 512×683 → 257 | 41 s | 3.8 s | **1.2 s** |
| greek_wine 1024×697 → 513 | ~4–5 min* | ~45 s | **5.6 s** |
(*estimated). v1 spent 79 % in scipy Sobel; v2 removes that (shared flip + band update).
Remaining cost: `build_axes` + `accumulate`, O(H·W) per cut, both directions.

### 3.6 Known behaviour: chaotic sensitivity
Greedy argmin over many near-tied cuts: a 1e-16 change of luminance can flip one choice and the
whole sequence diverges. Measured over 1e-9 input noise: face blur-L1 to 2013 result 0.029–0.041
(SC 0.064), teaser 0.062–0.079 (SC 0.075). Quality is a distribution, not a point — the E2E paper
claims are therefore judged on the median of 7 runs; single-image metrics (e.g. 26–28/36 "beats SC")
move by a few cases between equivalent builds.

### 3.7 Tests
| file | what |
|---|---|
| `test_core.py` | interpolation areas, integer cut = column delete, constant image, full-width straight edge keeps slope (SC fails) |
| `test_fast.py` | numba ≡ numpy cut-by-cut; incremental state ≡ full recompute; full resize equal on paper images |
| `test_e2e_paper.py` | A: paper claims on Fig.2 / Fig.1 teaser / Fig.12 face vs 2013 outputs (median of 7 runs, each also asserted to fail for SC); B: exact golden outputs (`UPDATE_GOLDEN=1` to refresh) |
Mutation-checked: `xi1=0.02` and `α=0` (no tangent following) both fail the E2E suite.

### 3.8 Extension points / known debt
* Energy is hard-wired to `|∇F|` and shared with the tangent field → to add saliency, split
  "direction field" from "cost field" (pass `e` separately to `accumulate`; numba already takes it).
* Two implementations (numpy + numba) must stay in lock-step; `test_fast.py` is the guard.
* `link_energy="segmax"` and `grad_mode="rgb"` exist only in numpy.
* Not a pip package; scripts insert the repo into `sys.path` and some use
  `~/Projects/ImageResizing2013` data paths. Enlarging is not defined by the paper.
