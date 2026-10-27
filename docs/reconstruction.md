# ASTRID sightlines, external Dachshund, multipoles and S/N

This workflow uses the existing `dachshund.exe` installed on M3. It does not
replace, vendor, or implement the Wiener solver. The old hard-coded script in
`reconstruction/run.py` is now an import-safe CLI entry point. Its ASTRID reader
still uses `fake_spectra.plot_spectra.PlottingSpectra`, z sightlines (axis=3),
ckpc/h to cMpc/h conversion, and z=j*L/N pixel coordinates by default.

## Run the measurement

From an M3 login node, activate your existing scientific environment with
fake_spectra and the compiled Dachshund executable already available:

```bash
python -m pip install -e '.[io,test]'
python -m pytest -q
export DACHSHUND=/users/jiawenfan/work/lya_CO/dachshund/dachshund.exe
# Replace this with the actual normalization used to create the true map.
export MEAN_FLUX=YOUR_TRUE_MAP_MEAN_FLUX
sbatch scripts/reconstruction_compare.sbatch
```

MEAN_FLUX is deliberately required: estimating it from a selected skewer subset
would silently change the delta_F convention. Use the true map's tau_scale too
(default 1). The map dataset is already delta_F and is never converted again.
Default input paths still point to z2.5; verify CO and spectra/map redshifts,
axis order, origin, and redshift-space convention before interpreting results.
To use different inputs, run `python scripts/reconstruction_compare.py --help`.
`--pixel-origin 0.5` is available only for datasets whose native LOS pixels are
at cell centers. No orientation is chosen by maximizing correlation.

Dachshund takes binary float64 rows (x,y,z,sigma_deltaF,deltaF), no header.
The configuration and output layout were checked against upstream
[app/dachshund.cc](https://github.com/caseywstark/dachshund/blob/master/app/dachshund.cc)
and [lib/util.cc](https://github.com/caseywstark/dachshund/blob/master/lib/util.cc).
The output is C-order, z fastest, at cell centers. The noise column is a
standard deviation. With no `--noise-seed`, no measurement noise is added;
sigma=.02 is still used as an assumed uncertainty/regularizer by the solver.
Check `dachshund/dachshund.log` for PCG convergence. The adapter rejects the
upstream max-iteration warning even when a map was written; inspect other
solver diagnostics as well. A finite map alone does not prove convergence.

The batch request is 4 CPUs, 64 GB, 24 hours. nmesh=250 keeps kmax=1 below half
Nyquist for the 250 cMpc/h box. Dachshund's map multiplication scales as
Nmap*Npixels; this full-box job can be expensive. The request is not a runtime
guarantee. First validate a smaller case with consistent nmesh/kmax. Reduce
sightlines/LOS pixels for speed only if the resulting sampling is appropriate;
the script rejects kmax above the sampled LOS Nyquist. No AI software runs on M3.

## Measurements and numerical controls

All methods share bins and LOS=z, with ell=0,2,4:

1. CO x full noiseless delta_F: full-map FFT ground truth.
2. CO x Dachshund reconstruction: full-grid FFT, retaining its transfer function.
3. CO x the exact same selected/noisy sightline pixels: configuration-space
   weighted pairs at their actual continuous positions, not grid-deposited spectra.

FFT uses V/Ncell^2 and full +/-k counts. The pair normalization is
V/(N_CO_samples*N_Ly_pixels), appropriate for uniform random CO cells and
uniformly selected uniformly sampled sightlines. The pair kernel is
(2ell+1)(-1)^(ell/2) j_ell(kr) L_ell(mu), with a cosine cutoff at rmax.
Pairs are accumulated into radial bins (default dr=.1 cMpc/h). Halve dr and
increase `--co-samples` to check convergence. Default 20,000 CO cells is Monte
Carlo volume quadrature and adds integration noise; it does NOT use every CO
voxel. Set co-samples to nmesh^3 for all voxels, at much higher cost. Selected
CO IDs are saved and held fixed across noise realizations. The pair kernel is
validated against the existing explicit pair estimator for ell=0,2,4.

The comparison intentionally contains different responses: the FFT ground
truth has no radial cutoff; pairs have the configured cutoff; reconstruction
has Wiener smoothing and nonperiodic boundaries in the upstream package.
The simulation pair distances use periodic minimum images. Differences are
not automatically reconstruction bias or gains in information. No survey
window deconvolution, sparse aliasing correction, HCD detection, or arbitrary
survey selection-function correction is claimed.

## Outputs

`results/reconstruction-JOBID-reference/` contains:

- multipoles.png and multipoles.csv: all three cross-power methods.
- multipoles.npz: the numerical results with common bins/modes.
- auto_spectra.npz: CO/truth/reconstruction auto multipoles and selected-skewer
  1D spectrum (two-sided P1D convention, k in h/cMpc, P1D in cMpc/h).
- truth_gaussian_sn.png/csv/json/npz: full-map Gaussian sample-variance S/N,
  including ell-ell covariance within k bins, estimated using eight |mu| cells.
  Low-k angular-rank deficiencies are recorded; total uses identifiable modes.
- reconstruction.npy, sightlines.npz, Dachshund inputs/config/output/log,
  sightline_ids.csv, pair_co_ids.csv, config.json including executable hash.

Keep binary cubes and pixel files on HPC; commit lightweight plots/CSVs/JSON.

## Reconstruction and pair S/N

A single simulation realization cannot supply their true covariance. The
pipeline calculates it from matching ensembles rather than using the complete
map mode-count formula for the sparse estimator.

For **conditional pixel-noise S/N**, hold the ASTRID fields, selected sightlines,
and CO quadrature fixed and add independent Gaussian pixel noise with sigma=.02.
Run the reference first, check its accuracy and cost, then submit repeated jobs:

```bash
# 128 realizations, at most 2 running at once; same exported MEAN_FLUX.
sbatch --array=0-127%2 scripts/reconstruction_compare.sbatch
# Replace REF_JOB and ARRAY_JOB with the returned job IDs.
python scripts/ensemble_sn.py \
  --reference results/reconstruction-REF_JOB-reference \
  --realizations results/reconstruction-ARRAY_JOB-[0-9]* \
  --kind noise-only --output results/sn-ARRAY_JOB
```

Each realization runs the actual external Dachshund solver on its noisy pixels
and counts pairs using those same pixels. This can be a large allocation: do
not launch the ensemble before measuring one job's cost. More realizations
are preferable to the minimal 128. Forty k bins and three multipoles give up
to 120 data entries. The code requires Nmock>Ndata+2, checks positive definite
covariance, and applies the Gaussian Hartlap inverse-covariance correction.
No pseudo-inverse or diagonal fallback hides an insufficient ensemble.

Output includes per-method covariance, per-bin S/N=abs(P)/sqrt(C_ii), and
joint S/N=sqrt(P^T C^-1 P), retaining k/ell correlations. `sn_multipoles.png`
and `sn_summary.json` label the covariance type. Noise-only S/N excludes
cosmic variance, random sightline-selection variance, and CO quadrature
variance; it cannot be ranked against the ground-truth sample-variance S/N
as a total survey forecast. No truth noise-only S/N is reported because that
fixed noiseless map has zero variance in this ensemble.

For a comparable **total mock S/N for all three methods**, run independent
matched CO/Ly-alpha simulation/mock realizations through the same pipeline
and call `ensemble_sn.py --kind independent-mocks` with their result folders.
All three then use their own empirical covariance from the same ensemble.
Repeated noise draws on the same fields are rejected as independent mocks.
The user must supply physically representative independent field realizations;
the pipeline does not fabricate cosmic variance from one ASTRID cube.

## Validation limits

Unit tests cover flux normalization, 1D Parseval normalization, Fourier
monopole equivalence and anisotropic multipoles, radial-pair agreement,
ASTRID reader conventions, covariance/S/N, and the external executable file
contract using a clearly identified stub. The stub is not a reconstruction
solver. Actual ASTRID data and your compiled Dachshund executable are only on
HPC, so their convergence and full-volume results must be verified there.
