"""Combine LiLa Ly-alpha power, Cartesian CO power, and FFTPower cross power.

Use with noiseless, spatially aligned CO and delta_F maps of the SAME volume.
Units: box cMpc/h; CO microK; k h/cMpc; noise powers in the corresponding
field-squared * (cMpc/h)^3 units. Run in your existing LiLa environment.

This is a spherical, diagonal Gaussian forecast. Noise inputs must describe
these same map fields (including any smoothing). It does not model a survey
mask, reconstruction transfer function, anisotropic beam, or non-Gaussian
covariance. Do not pass already-noisy auto spectra and add their noise again.

Notebook example (Tco and lya_mock already loaded):
    from co_lya_sn import combine_power_and_sn
    result = combine_power_and_sn(Tco, lya_mock, noise_co=0, noise_lya=0)
    np.savez("co_lya_sn_results.npz", **result)
The zeros give sample-variance-only S/N. Replace with your noise powers for
an observational forecast. CO white noise: N_CO = sigma_voxel**2 * V_voxel,
where sigma_voxel is in microK and V_voxel in (cMpc/h)^3 on that noise grid.
"""
from copy import copy
import numpy as np


def rebin_mean(field, shape):
    """Average integer-sized blocks, one output x slab at a time."""
    shape = np.asarray(shape, dtype=int)
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
    Nco = noise_co(k) if callable(noise_co) else noise_co
    Nlya = noise_lya(k) if callable(noise_lya) else noise_lya
    Nco = np.broadcast_to(np.asarray(Nco, dtype=float), k.shape)
    Nlya = np.broadcast_to(np.asarray(Nlya, dtype=float), k.shape)
    if np.any(Nco < 0) or np.any(Nlya < 0) or volume_ratio <= 0:
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


def combine_power_and_sn(Tco, lya_mock, noise_co, noise_lya,
                         dk=0.05, kmin=0.03, kmax=1.0,
                         final_transpose=(1, 0, 2), volume_ratio=1.0):
    """Return matched spectra and S/N, starting with the notebook's lya_mock.

    final_transpose is applied AFTER lya_mock.transpose. Its default reproduces
    the extra swapaxes(1, 0) in cross_test.ipynb. If your LiLa map is already
    aligned with CO, pass (0, 1, 2). Geometry/alignment must be physically verified.

    noise_co/noise_lya are REQUIRED: use 0 explicitly for a sample-variance-only
    calculation, or supply effective noise powers on these map fields.
    """
    from lila.stats import Stats
    from nbodykit.lab import ArrayMesh, FFTPower
    from lim_mocker_cartesian import power_spectrum

    if kmin <= 0:
        raise ValueError("Use kmin > 0 to exclude the mean (zero Fourier mode)")
    box = np.asarray(lya_mock.boxsize, dtype=float)
    Tco = np.asarray(Tco)
    dF = lya_mock.get_map_noiseless()
    dF = np.transpose(dF, final_transpose)
    dF = rebin_mean(dF, Tco.shape)
    if lya_mock.flux_mode:
        dF = dF / lya_mock.get_mean_flux(z=lya_mock.z) - 1.0
    dF = np.ascontiguousarray(dF - dF.mean(), dtype=np.float32)

    # Use the SAME rebinned Ly-alpha field in LiLa's auto and the cross power.
    # A shallow copy avoids modifying the original lya_mock and its cached map.
    matched_lya = copy(lya_mock)
    matched_lya.noiseless_map = dF
    matched_lya.Nmesh = list(dF.shape)
    matched_lya.flux_mode = False
    k_nyquist = np.min(np.pi * np.asarray(Tco.shape) / box)
    if kmax > 0.5 * k_nyquist:
        raise ValueError("Use kmax <= half the smallest grid Nyquist frequency")
    settings = dict(mode="1d", BoxSize=box, kmin=kmin, kmax=kmax, dk=dk)

    lya_stats = Stats(mock_lya=matched_lya, kmin=kmin, kmax=kmax, dk=dk)
    auto_lya = lya_stats.get_lya_pk(mode="1d").power
    edges = np.asarray(auto_lya.edges["k"])
    auto_co = power_spectrum(Tco, box, k_edges=edges)
    cross = FFTPower(first=ArrayMesh(Tco - Tco.mean(), BoxSize=box),
                     second=ArrayMesh(dF, BoxSize=box), **settings).power

    np.testing.assert_allclose(cross.edges["k"], edges)
    np.testing.assert_array_equal(auto_co["nmodes"], cross["modes"])
    np.testing.assert_array_equal(auto_lya["modes"], cross["modes"])
    # Counts must agree; equal array lengths alone do not establish matching bins.
    result = signal_to_noise(
        np.asarray(cross["k"]), auto_co["Pk"], auto_lya["power"].real,
        cross["power"].real, cross["modes"], noise_co, noise_lya, volume_ratio)
    result["k_edges"] = edges
    result["boxsize_cmpch"] = box
    result["volume_ratio"] = volume_ratio
    for name in ("CO", "F", "cross"):
        print(f"{name} total S/N = {result['sn_' + name + '_total']:.3f}")
    return result


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
