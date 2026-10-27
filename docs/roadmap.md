# Pipeline capability roadmap

## Current inputs and flow

Existing Cartesian CO temperature cube + ASTRID noiseless Ly-alpha map +
ASTRID fake_spectra optical depths/positions -> common units and grid ->
full-map FFT baseline, external Dachshund reconstruction, and sparse pair
measurements -> auto/cross multipoles, diagnostic plots and covariance/S/N.

## Implemented

| Stage | Capability | Current boundary |
|---|---|---|
| CO | Load map_cube NPZ, block average, auto/cross power | Original Li-model map maker still missing; supply an existing cube |
| Ly-alpha | Read full HDF5 map and actual z-directed fake_spectra sightlines | Requires matching volume, orientation, redshift and mean-flux normalization |
| Flux | Explicit F/mean(F)-1 conversion or already-delta_F input | Default supplied ASTRID map is delta_F; no double normalization |
| Sampling | Seeded sightline selection, LOS stride, optional Gaussian pixel noise | Uniform selection and equally spaced LOS pixels |
| Reconstruction | Write Dachshund files, run existing executable, read center-aligned map | No replacement solver; actual M3 binary/data run still to validate |
| Spectra | Full-map and sightline 1D power; auto/cross multipoles ell=0,2,4 | Multipoles retained on common Fourier bins; voxel windows retained |
| Sparse cross | Radially binned pairs at actual positions | CO-cell sampling defaults to 20,000; numerical convergence needed |
| Sparse auto | Distinct-sightline pairs, excluding same-LOS contributions | Anchor sampling and integral constraint; no survey deconvolution |
| Ground truth | Full noiseless-map FFT; older full/sparse matched-window controls retained | Periodic simulation geometry |
| Plotting | Three-method cross comparison; CO auto/Ly-alpha auto/cross panels; S/N plots | Produces scientific results only after data runs |
| S/N | Gaussian full-map sample variance; empirical cross covariance from ensembles | Noise-only and independent-field covariance explicitly distinguished |
| Reproducibility | Config, selected IDs, random seeds, executable hash, code revision | Large maps remain on HPC |

## Validation status

35 local tests passed after adding full-map/sparse auto-cross options.
Small numerical tests cover normalization, multipoles, explicit pair agreement,
input conversion, covariance, and process/file handling. Synthetic command-line
smoke tests used a stub executable, not the real Dachshund solver. Prior user
HPC comparisons validated the older monopole workflow; they do not validate
the newly added real-position reconstruction/multipole pipeline.

## Next milestones

1. Run the new branch on M3; verify mean flux, coordinate origin, input redshifts
   and PCG convergence. Compare selected spectra against the corresponding map.
2. Check numerical convergence versus radial bin width, CO-cell sample count,
   auto anchor count, LOS stride, nmesh and reconstruction tolerance.
3. Quantify radial-window, reconstruction-transfer and boundary effects before
   interpreting differences between the three methods as information gains.
4. Run noise ensembles for conditional S/N; supply independent matched mock
   fields for total S/N comparable across all methods. Sparse-auto covariance
   is not yet included in the cross-S/N ensemble command.
5. Restore the original CO Li-model map maker, add observational survey/beam/
   continuum/HCD treatment as needed, and extend geometry/selection corrections.

Use docs/reconstruction.md for commands and limitations. No AI agents or API
calls are run on M3.
