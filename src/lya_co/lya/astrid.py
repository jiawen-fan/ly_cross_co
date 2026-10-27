"""ASTRID readers; positions cMpc/h, flux fluctuations dimensionless."""
from pathlib import Path
import numpy as np
import h5py
from lya_co.co.co_ly_sn import rebin_mean


def flux_to_delta(flux, mean_flux):
    if not np.isfinite(mean_flux) or mean_flux <= 0:
        raise ValueError('mean_flux must be positive')
    return np.asarray(flux) / mean_flux - 1


def delta_to_flux(delta, mean_flux):
    if not np.isfinite(mean_flux) or mean_flux <= 0:
        raise ValueError('mean_flux must be positive')
    return (1 + np.asarray(delta)) * mean_flux


def load_maps(co_path, map_path, nmesh):
    with np.load(co_path) as f:
        co = np.asarray(rebin_mean(f['map_cube'], (nmesh,)*3))
    with h5py.File(map_path, 'r') as f:
        truth = np.asarray(rebin_mean(f['map'], (nmesh,)*3))
    return co, truth


def read_sightlines(path, count, seed, mean_flux, tau_scale=1., stride=5,
                    sigma=0.02, noise_seed=None, pixel_origin=0.):
    """Read real fake_spectra skewers, axis=3; no gridding of pair positions.

    mean_flux must match the noiseless map. pixel_origin=0 reproduces the old
    script's z=j*L/N convention; choose 0.5 only if the data use pixel centers.
    sigma is the assumed delta_F standard deviation. noise_seed=None adds none.
    """
    from fake_spectra.plot_spectra import PlottingSpectra
    path = Path(path).resolve()
    sp = PlottingSpectra(num=1, base='./', savedir=str(path.parent),
                         savefile=path.name, res=None)
    box = float(sp.box) / 1000.
    available = np.flatnonzero(np.asarray(sp.axis) == 3)
    if not 1 <= count <= len(available) or stride < 1 or sigma <= 0:
        raise ValueError('Invalid sightline count, stride or sigma')
    ids = np.sort(np.random.default_rng(seed).choice(available, count, replace=False))
    tau = sp.get_tau(elem='H', ion=1, line=1215)
    nz = tau.shape[1]
    if nz % stride:
        raise ValueError('Stride must divide native LOS pixels for uniform quadrature')
    values = flux_to_delta(np.exp(-tau_scale * tau[ids, ::stride]), mean_flux)
    if noise_seed is not None:
        values += np.random.default_rng(noise_seed).normal(0, sigma, values.shape)
    xy = np.asarray(sp.cofm)[ids, :2] / 1000.
    z = (np.arange(nz)[::stride] + pixel_origin) * box / nz
    xyz = np.empty((count, len(z), 3))
    xyz[..., :2] = xy[:, None, :]
    xyz[..., 2] = z
    if np.any(xyz < 0) or np.any(xyz >= box) or not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite flux or out-of-box sightline coordinates')
    return dict(positions=xyz.reshape(-1, 3), delta=values.ravel(), ids=ids,
                values=values, box=box, dz=box/nz*stride, sigma=sigma)
