# Full-map versus sparse-sightline cross power

Run `sbatch scripts/sparse_pairs.sbatch` from the repository root on an M3
login node, with your scientific Python environment active. The request remains
4 CPUs, 64 GB and 4 hours, nmesh=250, 1000 sightlines, seed=42,
kmax=1 h/cMpc, 40 common bins, cosine radial cutoff=80 cMpc/h.
No AI software or API calls are used on M3.

The input CO NPZ uses `map_cube` (microK); Ly-alpha HDF5 uses `map`
(already delta_F). Both must cover the same aligned periodic box, in the same
axis order. The code block-averages each to nmesh; it never upsamples or removes
the voxel window. Full-map reading is slabwise during downsampling.
The supplied Ly-alpha path says z2.5: matching redshifts/coordinates must be
verified from the datasets, not changed by setting a plotting label.

## Curves

- `full_fft`: original complete-map discrete Fourier shell estimator.
- `sparse_pair_monopole`: CO full grid crossed only with selected Ly-alpha
  columns, computed by exact periodic grid compression of the weighted pair sum.
- `full_pair_monopole`: same radial window/Bessel estimator with all columns;
  compare this to sparse pairs to isolate sampling effects.
- `full_windowed_fft`, `sparse_windowed_fft`: Fourier shell averages with the
  same real-space cosine window; these distinguish window and shell-averaging
  effects from the radial Bessel result evaluated at the mean k in each shell.

All use identical k edges, geometry, and CO input. FFT normalization is V/Ncell^2;
pair weights are dV_CO*dV_F/p divided by V, with p=Nselected/(nx*ny).
The selected field is centered on its sample mean and weighted by 1/p.
Zeros in its computational accumulator represent missing samples, not measured
zero flux. Selected columns are finite-area block averages along axis 2, not
arbitrary-position pencil-beam spectra from fake_spectra. No reconstruction,
survey-window correction, or sparse covariance model is included.

## Outputs

Each run creates a new `results/sparse-JOBID/` directory:

- `comparison.png`: all five cross spectra, linear and signed logarithmic panels.
- `full_map_diagnostics.png`: complete-map CO and Ly-alpha auto power and r(k).
- `spectra.csv`: shared k, full +/-k mode counts, cross spectra, autos, r.
- `spectra.npz`: the same numerical results plus bin edges and selection metadata.
- `sightlines.csv`: selected transverse grid indices, retained with lightweight results.
- `config.json`: input paths, scales, means, seed, selection probability, units,
  code revision, and redshift caveat.

Empty Fourier shells are NaN. Mode counts and r refer to the complete periodic
map: they do not supply sparse-pair uncertainties or an observational S/N.
Do not use the sparse accumulator auto power as a physical Ly-alpha auto spectrum.
The sample-mean subtraction can cause a sampling-dependent integral constraint.

Reusable functions are in `lya_co.estimator.full_sparse`; the command-line
loader and plotting functions are in `scripts/sparse_pairs.py`.
The small-grid tests compare compressed sparse pairs with explicit weighted
pairs, check all-sightline agreement, and verify that changing unselected
Ly-alpha pixels changes only the complete-map baseline.
