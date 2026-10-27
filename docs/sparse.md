# Sparse mock sightline experiment

Submit `sbatch scripts/sparse_pairs.sbatch` from the repository root on an
allowed M3 login host, with your scientific Python environment active.
Requests 64 GB, 4 CPUs, 4 hours on htc. Cluster resource availability is not
verified here. More memory supports the higher-resolution grid; NumPy FFTs
are not guaranteed to use all four CPUs.

Defaults: periodic L=250 cMpc/h, 250^3 grid, 1000 random full z-columns without
replacement, seed 42, kmax=1 h/cMpc, cosine separation taper ending at 80 cMpc/h.
Mean transverse spacing is about 7.9 cMpc/h. Each sightline has finite transverse
voxel width: block-averaged native map cells, not a pencil beam. Only selected
Ly-alpha column blocks are read. No flux renormalization, reconstruction or
pixel noise is added. Physical alignment of input axes is assumed.

The weighted sparse pair monopole uses dV for CO and dV/p for selected Ly-alpha
pixels, with p=Nsightlines/(Nx Ny). Means are removed from CO and from the
selected Ly-alpha sample. The correlation-grid implementation exactly compresses
this discrete pair sum; the masked field is a computational accumulator, not a
claim that unobserved flux is zero. Tests compare against explicit sparse pairs.

Random selection weights do not supply a general survey correction. Sample-mean
subtraction can introduce an integral-constraint effect; sightline correlations,
noise, aliasing and covariance are not calibrated. Continuous-angle pair power
and discrete Fourier shells need not match. The fixed radial taper changes both.
This version supports grid-aligned mock sightlines only, not arbitrary fake_spectra
coordinates. It is a higher-k numerical experiment, not a validated forecast.

The supplied Ly-alpha filename indicates z=2.5; CO redshift is unverified.
Outputs explicitly record the redshift as unverified, not z=2.6. Verify metadata
before physical interpretation. No redshift transformation is performed.

Outputs: results/sparse-JOBID/{comparison.png,spectra.csv,spectra.npz,config.json}.
NPZ includes selected transverse indices; config includes seed and probability.
Default kmax is below half Nyquist (1.57 h/cMpc), but this alone does not validate
small-scale aliasing or painting-window effects. Re-run with different seeds and
sightline counts to assess sampling variation. Full-map baseline remains available
through scripts/compare_estimators.py with --nmesh 250 --skip-pairs --kmax 1.
