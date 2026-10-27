# Matched FFT versus HIPSTER-inspired pair estimator

This combined branch contains the changes from `fix/matched-cross-power`
(30fd240) and `feature/hipster-inspired-cross-estimator` (0c11768), plus the
comparison driver, as one commit on top of the initial main commit b880d2a.
The original branches are retained. No HPC files were changed remotely.

## What is being compared

| Saved array | Meaning | Expected agreement |
|---|---|---|
| `fft_cross` | Matched auto/cross estimator, discrete Fourier-shell mean | Baseline |
| `fft_windowed` | Multiply periodic cross-correlation by the pair separation window, then FFT and bin | Equals baseline when the window is unity |
| `radial_reference` | Sum that same correlation grid against j0(k r) | Equals the old pair monopole |
| `pair_cross` | Original explicit sum of all CO/Lyα voxel pairs | Equals radial reference, not necessarily FFT shells |

Let N be the number of voxels and dV=V/N. We compute
`xi[d] = sum_x a[x] b[x-d] / N` from mean-subtracted fields.
The old pair monopole is exactly `dV sum_d xi[d] W(d) j0(k |d|)`
for a complete periodic grid with minimum-image displacements. Its k values
are the mean magnitudes in the FFT bins, not a bin integral of the Bessel
kernel. The FFT averages discrete lattice wavevectors instead of a
continuous angular sphere. This distinction remains even without tapering;
ratios near a zero crossing are not a useful correctness test.

The comparison checks the two identities above; it does not assert that
`pair_cross == fft_cross`. Inspect the four curves to distinguish estimator
conventions from numerical errors. A separation taper changes the spectrum
and can mix signs or modes; it is not an observational survey correction.

The driver measures the monopole only. The old module's multipole support
is retained. Both fields must be complete periodic, aligned grids in the
same physical volume, at the same redshift and with the same RSD convention.
No automatic permutation or redshift correction is made. Lyα is already
delta_F; CO is microK. All cross powers are microK (cMpc/h)^3.

## M3 setup

Use an existing scientific Python **3.9 or newer** environment. These are
ordinary NumPy/SciPy computations, with no AI packages, credentials or API
calls. Install dependencies through your usual permitted HPC method. From
the repository root, with that environment active:

```bash
python -m pip install -e '.[io,test]'
sbatch scripts/compare_estimators.sbatch
```

The batch script requests standard-s, one task, four CPUs and 16 GB for 30
minutes. Adjust allocation/account/partition to your M3 permissions. NumPy
FFT and the pair loop are not fully multithreaded; allocating four CPUs does
not guarantee four busy CPUs. The script runs the full test suite and two
synthetic comparisons, untapered and cosine-tapered. It stops on failures.

Then submit existing ASTRID maps:

```bash
sbatch scripts/compare_estimators.sbatch \
  /path/to/co_z2.6.npz \
  /path/to/lya_delta_f_z2.6.hdf5
```

Replace these placeholders with actual **z=2.6** maps for both fields. The
Lyα dataset name defaults to `map`; CO defaults to `map_cube`. The previous
z=2.5 Lyα file is not appropriate for this run. No z=2.6 Lyα path has yet
been verified. The CLI defaults to `--redshift 2.6` and records it in the
configuration; this is an input declaration, not a redshift conversion.
Verify matching geometry, origin and RSD convention as well as snapshot.

The real-data runs average the *whole box*, not a spatial cutout, to 10^3
voxels. Native dimensions must be multiples of 10. The pair loop then has
10^6 pairs. kmax defaults to half the smallest Nyquist frequency: about
0.0628 h/cMpc at this resolution for L=250 cMpc/h. This is a low-resolution
consistency test, not an ASTRID high-k measurement. Block-average windows
are retained in both methods.

## Outputs and larger grids

Each new `results/comparison-JOBID-*` directory contains:

- `comparison.png`: the four signed power curves.
- `spectra.csv` and `spectra.npz`: k, mode counts and all comparisons.
- `config.json`: arguments, input paths, seed, Git revision, elapsed time
  and numerical identity checks.

Expected: `pair_vs_radial_pass: true`; untapered runs also have
`untapered_fft_pass: true`. These test implementation consistency, not
physical calibration, uncertainty or S/N. Empty bins remain NaN.
Output directories must be new so earlier results cannot be overwritten.

For a more resolved periodic grid use the exact correlation-grid compression
and skip the explicit pair loop, e.g. inside a batch allocation:

```bash
python scripts/compare_estimators.py --co CO.npz --lya LYALPHA.hdf5 \
  --nmesh 50 --skip-pairs --output results/comparison-fine
```

50 divides both 250 and 1000. This is mathematically the old monopole on a
complete grid, evaluated without enumerating all pairs; the saved
`pair_cross` values are NaN because the old loop was skipped. It does not
accelerate the general sparse-sightline problem. HDF5 Lyα is read in slabs;
compressed NPZ CO loading still decompresses the native cube in memory.
Increase the memory allocation if your CO file contains a native 1000^3 cube.
