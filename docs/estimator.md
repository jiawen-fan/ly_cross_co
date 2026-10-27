# CO × Lyα pair-based cross-power estimator (prototype)

This implements a **small-catalog reference estimator**, inspired by the
configuration-space pair summation in [HIPSTER](https://github.com/oliverphilcox/HIPSTER).
It is **not** a reproduction of HIPSTER's optimized estimator and **not** an
unwindowed, survey-corrected measurement for sparse Lyα sightlines.

## Formula and conventions

For two real-valued sampled fields `a` (CO temperature in µK) and `b` (Lyα
flux contrast `δ_F`), at positions `x_i` and `y_j`, the estimator is

\[
\widehat P_\ell(k) = \frac{(2\ell+1)(-1)^{\ell/2}}{V}
 \sum_{i,j} v_i^{(a)} v_j^{(b)} a_i b_j\,
 W(r_{ij})\,C(r_{ij},\mu_{ij})\,j_\ell(kr_{ij})
 \mathcal L_\ell(\mu_{ij}), \quad \ell=0,2,4,\ldots
\]

Here `r_ij` is the separation, `mu_ij` is its cosine relative to the chosen
line of sight, `v` are **integration volumes** (not inverse-noise weights),
`W` is a configurable radial taper, and `C` is an *optional user-supplied*
geometry correction. Coordinates are in comoving Mpc/h, `k` is in h/comoving
Mpc, and the cross-power units are µK × (comoving Mpc/h)^3. With `subtract_mean=True`,
the code subtracts volume-weighted sample means before counting pairs.

For periodic boxes the separation follows the minimum-image convention. For
nonperiodic surveys, the code retains only the supplied measured locations.
This **does not remove mask convolution automatically**. The optional `C(r,mu)`
requires calibration against an independent survey selection / random pair
model; passing a constant does not generate a legitimate survey correction.
A truncation or taper in `r` also induces its own Fourier-space window.

## Quick local test (no simulation files required)

```bash
PYTHONPATH=src python -m pytest -q tests/test_pair_cross.py
```

Example using a tiny grid (do *not* supply full ASTRID cubes to this backend):

```python
import numpy as np
from lya_co.estimator.pair_cross import Geometry, grid_to_samples, estimate_cross_power

geometry = Geometry(boxsize=(250., 250., 250.), periodic=True, los_axis=2)
co_cube = np.random.default_rng(3).normal(size=(4, 4, 4))
lya_cube = -co_cube
lya_mask = np.random.default_rng(4).random((4, 4, 4)) < 0.5
co_pos, co, co_vol = grid_to_samples(co_cube, geometry)
lya_pos, lya, lya_vol = grid_to_samples(
    lya_cube, geometry, mask=lya_mask, selection_probability=0.5
)
result = estimate_cross_power(
    co_pos, co, lya_pos, lya, k=np.array([0.1, 0.2]),
    geometry=geometry, r_max=60., window="cosine", multipoles=(0, 2),
    co_sample_volumes=co_vol, lya_sample_volumes=lya_vol,
)
print(result.power)  # shape: (2 multipoles, 2 k bins)
```

The inverse selection probability in this toy example assumes *independent
random voxel sampling*. Real Lyα sightline selection is correlated along
skewers; its window, aliasing, noise, and mode coupling need an explicit
survey treatment. A reconstructed Lyα cube is a different input model from
raw skewers and must be validated separately.

## Deliberate limitations and next work

- **O(N_CO × N_Lyα)** brute-force; `max_pairs` fails fast for large input.
  Replace the inner pair loop with a KD-tree / Corrfunc or HIPSTER-style backend.
- No R/R, D/R random pair counts, full HIPSTER `Φ` calibration, mode-coupling
  matrix, anisotropic survey corrections, shot-noise treatment, or covariance.
- No direct FFT-to-pair agreement expected at fixed discrete `k`: Fourier
  shell-averaging and radial-window conventions differ. Compare synthetic
  fields, matched windows, and binned predictions before using scientific data.
- Do not treat sparse sightline `1/p` weights as sufficient to make an
  unbiased observational cross-power estimate. Build the survey selection
  model and validate against known synthetic fields first.
- This module has **no network/AI dependencies** and is only intended to
  run after ordinary scientific review and GitHub synchronization.
