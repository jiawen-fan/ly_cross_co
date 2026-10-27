"""Measure matched Cartesian CO/Ly-alpha auto and cross power and S/N.

Use with noiseless, spatially aligned CO and delta_F maps of the SAME volume.
Units: box cMpc/h; CO microK; k h/cMpc; noise powers in the corresponding
field-squared * (cMpc/h)^3 units. Run in your existing LiLa environment.

This is a spherical, diagonal Gaussian forecast. Noise inputs must describe
these same map fields (including any smoothing). It does not model a survey
mask, reconstruction transfer function, anisotropic beam, or non-Gaussian
covariance. Do not pass already-noisy auto spectra and add their noise again.

Notebook example (Tco and lya_mock already loaded):
    from lya_co.co.co_ly_sn import combine_power_and_sn
    result = combine_power_and_sn(Tco, lya_mock, noise_co=0, noise_lya=0)
    np.savez("co_lya_sn_results.npz", **result)
The zeros give sample-variance-only S/N. Replace with your noise powers for
an observational forecast. CO white noise: N_CO = sigma_voxel**2 * V_voxel,
where sigma_voxel is in microK and V_voxel in (cMpc/h)^3 on that noise grid.
"""
import numpy as np


def rebin_mean(field, shape):
    """Average integer-sized blocks, one output x slab at a time."""
    shape = np.asarray(shape)
    if shape.shape != (3,) or np.any(shape <= 0) or np.any(shape != shape.astype(int)) or len(field.shape) != 3:
        raise ValueError("Input and target must be 3D with positive integer dimensions")
    shape = shape.astype(int)
    old_shape = np.asarray(field.shape)
    if np.any(old_shape % shape):
        raise ValueError("Ly-alpha dimensions must be integer multiples of CO dimensions")
    if np.array_equal(old_shape, shape):
        return field
    fx, fy, fz = old_shape // shape
    nx, ny, nz = shape
    result = np.empty(tuple(shape), dtype=np.float32)
    for i in range(nx):
        slab = field[i * fx:(i + 1) * fx]
        result[i] = slab.reshape(fx, ny, fy, nz, fz).mean(
            axis=(0, 2, 4), dtype=np.float64)
    return result


def signal_to_noise(k, Pco, Plya, Pcross, modes, noise_co, noise_lya,
                    volume_ratio=1.0):
    """Gaussian S/N using FULL Fourier mode counts (including +/- k).

    Noise: nonnegative scalar, array matching k, or callable evaluated at k.
    volume_ratio = survey volume / map volume; this is only a mode-count
    rescaling, not an explicit survey-window calculation.
    """
    k, Pco, Plya, Pcross, modes = [np.asarray(a, dtype=float)
                                   for a in (k, Pco, Plya, Pcross, modes)]
    if k.ndim != 1 or any(a.shape != k.shape for a in (Pco, Plya, Pcross, modes)):
        raise ValueError('Spectra and mode counts must be matching 1D arrays')
    if np.any(~np.isfinite(modes)) or np.any(modes < 0):
        raise ValueError('Mode counts must be finite and nonnegative')
    Nco = noise_co(k) if callable(noise_co) else noise_co
    Nlya = noise_lya(k) if callable(noise_lya) else noise_lya
    Nco = np.broadcast_to(np.asarray(Nco, dtype=float), k.shape)
    Nlya = np.broadcast_to(np.asarray(Nlya, dtype=float), k.shape)
    if np.any(~np.isfinite(Nco[modes > 0])) or np.any(~np.isfinite(Nlya[modes > 0])) or np.any(Nco < 0) or np.any(Nlya < 0) or not np.isfinite(volume_ratio) or volume_ratio <= 0:
        raise ValueError("Noise powers must be nonnegative and volume_ratio positive")
    Nm = modes * volume_ratio
    valid = ((k > 0) & (Nm > 0) & (Pco > 0) & (Plya > 0)
             & np.isfinite(k + Pco + Plya + Pcross + Nco + Nlya))
    k, Pco, Plya, Pcross, Nm, Nco, Nlya = [
        a[valid] for a in (k, Pco, Plya, Pcross, Nm, Nco, Nlya)]
    if k.size == 0:
        raise ValueError("No common valid bins remain")

    # Halo shot noise is already included in Pco; do not add it again.
    sigma_cross = np.sqrt(((Pco + Nco) * (Plya + Nlya) + Pcross**2) / Nm)
    sigma_co = np.sqrt(2 / Nm) * (Pco + Nco)
    sigma_lya = np.sqrt(2 / Nm) * (Plya + Nlya)
    sn_cross = np.abs(Pcross) / sigma_cross
    sn_co = Pco / sigma_co
    sn_lya = Plya / sigma_lya

    return dict(k=k, P_CO=Pco, P_F=Plya, P_CO_F=Pcross,
                r=Pcross / np.sqrt(Pco * Plya), modes=Nm,
                noise_CO=Nco, noise_F=Nlya,
                sigma_cross=sigma_cross, sigma_CO=sigma_co, sigma_F=sigma_lya,
                sn_cross=sn_cross, sn_CO=sn_co, sn_F=sn_lya,
                sn_cross_total=np.sqrt(np.sum(sn_cross**2)),
                sn_CO_total=np.sqrt(np.sum(sn_co**2)),
                sn_F_total=np.sqrt(np.sum(sn_lya**2)), valid_bins=valid)


def combine_maps_and_sn(Tco, delta_f, boxsize, k_edges, noise_co, noise_lya,
                        volume_ratio=1.0):
    """Aligned complete periodic maps -> matched spectra and Gaussian S/N.

    CO is microK and delta_f dimensionless. Noise powers must refer to the
    same filtered fields. Zero noise explicitly requests sample variance only.
    Sparse sightlines and masked maps require a different estimator.
    """
    from lya_co.estimator.power import matched_power
    p = matched_power(Tco, delta_f, boxsize, k_edges)
    result = signal_to_noise(p['k'], p['P_CO'], p['P_F'], p['P_CO_F'],
                             p['modes'], noise_co, noise_lya, volume_ratio)
    result.update(k_edges=p['k_edges'], boxsize_cmpch=p['boxsize_cmpch'],
                  volume_ratio=volume_ratio)
    return result


def combine_power_and_sn(Tco, lya_mock, noise_co, noise_lya,
                         dk=0.05, kmin=0.03, kmax=1.0,
                         final_transpose=(0, 1, 2), volume_ratio=1.0):
    """LiLa map adapter. All spectra use the same rebinned field and FFT.

    final_transpose is applied AFTER LiLa's own transpose; identity is the
    default. Verify axis order and origin from metadata, not correlation size.
    Block averaging retains its voxel window; it is not deconvolved.
    """
    if not np.all(np.isfinite([dk, kmin, kmax])) or dk <= 0 or not 0 <= kmin < kmax:
        raise ValueError('Require dk > 0 and 0 <= kmin < kmax')
    if sorted(final_transpose) != [0, 1, 2]:
        raise ValueError('final_transpose must be an axis permutation')
    box = np.broadcast_to(np.asarray(lya_mock.boxsize, dtype=float), (3,))
    box = box[list(final_transpose)]
    dF = np.transpose(lya_mock.get_map_noiseless(), final_transpose)
    dF = rebin_mean(dF, Tco.shape)
    if lya_mock.flux_mode:
        mean_flux = lya_mock.get_mean_flux(z=lya_mock.z)
        if not np.isfinite(mean_flux) or mean_flux <= 0:
            raise ValueError('Mean flux must be positive and finite')
        dF = dF / mean_flux - 1.0
    if kmax > 0.5 * np.min(np.pi * np.asarray(Tco.shape) / box):
        raise ValueError('Use kmax <= half the smallest grid Nyquist frequency')
    edges = np.arange(kmin, kmax, dk)
    edges = np.append(edges, kmax)
    return combine_maps_and_sn(Tco, dF, box, edges, noise_co, noise_lya,
                               volume_ratio)


def lya_sampling_noise(lya_mock, pixel_noise_1d=0.0):
    """Return a callable for an approximate spherical sparse-sightline noise.

    Uses LiLa's 1D spectrum: N_F(k_parallel) = dperp^2 * (P_1D + P_pixel_1D),
    averaged uniformly in mu. pixel_noise_1d is in cMpc/h for delta_F. Zero
    means sampling-only noise: lya_mock.sn is NOT used. This assumes uniform
    independent sightlines, isotropic signal within a shell, and negligible
    voxel filtering on the scales used; it is not reconstruction noise.
    For beam/LOS-dependent forecasts use a full (k, mu) covariance instead.
    """
    from lila.stats import Stats
    k1d, P1d = Stats(mock_lya=lya_mock).get_1D_lya_pk()
    mu = (np.arange(64) + 0.5) / 64

    def noise(k):
        kpar = np.asarray(k)[:, None] * mu[None, :]
        # LiLa also uses np.interp: values outside the table use endpoints.
        p = np.interp(kpar, k1d, P1d)
        return lya_mock.dperp**2 * (p + pixel_noise_1d).mean(axis=1)

    return noise

