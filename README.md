# Ly-alpha and CO Cross-Correlation

Research pipeline for cross-correlating Ly-alpha forest
observations and CO line-intensity maps using simulations.

## Main Components
- Ly-alpha mock spectra
- CO intensity mapping
- Ly-alpha tomographic reconstruction
- HIgh-k Power SpecTrum and bispectrum EstimatoR
- Auto- and cross-power spectra
- Numerical verification and testing

## Computation
Scientific computations run on SMU M3 HPC.
Large datasets are stored outside this repository.

## Development
Source code is version-controlled with GitHub.
AI-assisted development takes place outside HPC.

##Code structure:
Pipeline structure



CO Map making: 
CO mapmaking, CO power spectrum generating
Functions:
1. Read in simulation
 	Load in halo mass, positions, velocities information.
2. Assign sfr to each halo: 
using behroozi table, assign sft based on log(halo mass).
3. Sfr to luminosity:
	SFR -> IR -> CO(1-0) luminosity in Li 2016, consider adding Chung 2019
4. Real space map making:
	Map making, painting luminosity and converting to CO temperature.CIC(cloud in cell) optional.
5. Map saving:
	Save map in ‘npz’ or ‘.hdf5’ files.
6. Calculating auto power spectrum:
	Spherical auto/cross P(k); k in h/cMpc, P in (cMpc/h)^3.

Lyman Alpha map reconstruction:
In construction
Noise input for realistic Ly alpha (if needed later)
Reading astrid.
Running dachshund input 
Reconstruction out put, rebinning the map for cross correlation
Out the map


Lyman Alpha map post processing:
In construction
1. Converting from delt F to Flux, and vise versa
2. ? Masking HCD.
3. Calculate 1d power spectrum 
4. Calculated spherical average power spectrum
5. Calculated power spectrum with multipole moments.


HIgh-k Power SpecTrum and bispectrum EstimatoR on CO and Lyman Alpha:
1. use the map generated.
2. follow Hipster code, instead of auto cross, make cross between co and lyman alpha with sparse sightlines
3. Calculating auto power spectrum:
	Spherical auto/cross P(k); k in h/cMpc, P in (cMpc/h)^3.


Cross power spectrum and S/N
Calculating signal to noise ratio for CO and lyman alpha
Calculating the cross power spectrum with different input maps
Comparing results


## Working periodic-map cross spectrum

Install from the repository root: `python -m pip install -e .`.
Run tests with `PYTHONPATH=src python -m unittest discover -s tests -v`.
Open `scripts/cross_test.ipynb` for loading, block averaging, cross-power,
S/N, and plotting from existing CO and Ly-alpha maps. It needs h5py and
matplotlib in addition to NumPy.

```python
from lya_co.co.co_ly_sn import combine_maps_and_sn
result = combine_maps_and_sn(Tco, delta_f, boxsize, k_edges,
                             noise_co=0, noise_lya=0)
```

Both maps must share physical volume, origin, axes, voxel locations,
redshift and real/redshift-space convention. `delta_f` is already F/<F>-1;
do not normalize it a second time. No axis permutation is guessed.
`combine_power_and_sn` remains a LiLa map adapter, but all three spectra
now use one NumPy estimator, independent of nbodykit and the CO map maker.

The estimator uses P_ab = V/Ncell^2 Re[FFT(a) FFT(b)*], subtracts global
means, excludes k=0 and counts both +k and -k modes. Bins are [left,right).
It retains empty bins as NaN; S/N removes invalid bins and returns a
`valid_bins` mask relative to the original `k_edges`. For CO in microK and
Ly-alpha delta_F, CO auto, Ly-alpha auto and cross power have units
microK^2 (cMpc/h)^3, (cMpc/h)^3 and microK (cMpc/h)^3 respectively.

The Gaussian cross variance is
`((P_CO+N_CO)*(P_F+N_F)+P_CO_F**2)/N_modes`.
Zero noise requests sample variance only. Halo shot noise in the map is
retained; do not add it a second time. Block averaging and painting windows
are retained, with no deconvolution. The notebook limits k to half the
smallest grid Nyquist frequency. Spherical diagonal errors are an
approximation for anisotropic fields; masks and reconstructed fields need
window/response and covariance treatment before observational interpretation.

### Known incomplete components in the initial commit

- `src/lya_co/co/lim_mocker_cartesian.py` is an exact copy of `co_ly_sn.py`,
  not the original map maker. `run_astrid.py`, `run_peak.py`, and the example
  map-making notebook cannot run until the original Li-model implementation
  is restored. Do not use these to regenerate maps yet.
- A small-catalog HIPSTER-inspired pair prototype is included in
  `src/lya_co/estimator/pair_cross.py`; see `docs/estimator.md`. It is not the
  optimized HIPSTER estimator or a calibrated sparse-sightline survey
  correction. Do not zero-fill missing sightlines into the periodic FFT.
- Full ASTRID runs require the external HPC datasets and verification of
  map metadata. Synthetic tests here validate Fourier normalization, odd/even
  mode counts, negative cross-power, rebinning and the Gaussian formula.

## Compare the two estimators on M3

See [docs/comparison.md](docs/comparison.md) for the definitions, tests,
Slurm commands, ASTRID inputs, and output interpretation. The unified
entry point is `scripts/compare_estimators.py`; the batch script is
`scripts/compare_estimators.sbatch`. Both old and matched estimators are
preserved.

The comparison target is **z=2.6 for both fields**. Supply the actual z=2.6
CO and Lyα maps; the redshift argument does not transform z=2.5 data.

## ASTRID spectra and external Dachshund comparison

The Ly-alpha, reconstruction and spectra modules now provide ASTRID
fake_spectra input, explicit flux/delta_F conversion, LOS 1D power,
auto/cross multipoles (ell=0,2,4), and a file adapter that runs your existing
Dachshund executable. `scripts/reconstruction_compare.py` compares the
reconstructed map, actual sparse-sightline pairs and the noiseless ground
truth against the same CO map. `scripts/ensemble_sn.py` calculates empirical
covariance and S/N from matching realizations; the full-map Gaussian S/N is
saved separately. See [docs/reconstruction.md](docs/reconstruction.md) for
M3 commands, required mean-flux normalization, computational controls,
outputs and distinctions between noise-only and total mock covariance.
HCD masking and survey-response corrections are not implemented.

The auto/cross summary defaults to the full Ly-alpha map, following LiLa's
signal-map approach. Add `--lya-power-source both` (Slurm:
`export LYA_POWER_SOURCE=both`) to also measure sparse distinct-sightline
auto power and sparse CO cross power. Raw flux input is explicit via
`--lya-map-field flux`; the supplied ASTRID delta_F map uses the default
`delta`. Sampling-noise estimates are saved separately from measured signal.
