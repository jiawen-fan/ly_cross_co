"""Compare periodic FFT shells to the original spherical-Bessel pair sum.

The correlation-grid radial sum is an exact compression of all voxel pairs
for full periodic grids. It is not a sparse survey/window correction.
"""
import numpy as np
from scipy.special import spherical_jn
from .power import matched_power
from .pair_cross import Geometry, grid_to_samples, estimate_cross_power


def compare_estimators(co, delta_f, boxsize, k_edges, r_max=None,
                       window='tophat', run_pairs=True, max_pairs=2_000_000):
    """Return monopoles using identical mean-subtracted maps and geometry.

    fft_cross: ordinary discrete Fourier shell mean.
    fft_windowed: discrete shell mean after multiplying xi(r) by W(r).
    radial_reference: Vcell sum_r xi(r) W(r) j0(k*r), evaluated at shell k.
    pair_cross: original O(Ncell^2) pair sum, matching radial_reference.
    Units are field_CO * field_F * (cMpc/h)^3; k in h/cMpc.
    """
    co, delta_f = np.asarray(co), np.asarray(delta_f)
    p = matched_power(co, delta_f, boxsize, k_edges)
    box = p['boxsize_cmpch']
    if window not in ('tophat', 'cosine'):
        raise ValueError('window must be tophat or cosine')
    if r_max is not None and (not np.isfinite(r_max) or r_max <= 0):
        raise ValueError('r_max must be positive and finite')
    if window == 'cosine' and r_max is None:
        raise ValueError('cosine requires r_max')
    if run_pairs and co.size**2 > max_pairs:
        raise ValueError('Pair limit exceeded: reduce nmesh or use --skip-pairs')
    # xi[d] = (1/Ncell) sum_x a[x] b[x-d].
    fa = np.fft.fftn(co - co.mean(dtype=np.float64))
    fb = np.fft.fftn(delta_f - delta_f.mean(dtype=np.float64))
    xi = np.fft.ifftn(fa*fb.conj()).real / co.size
    cell = box / co.shape
    offsets = [np.fft.fftfreq(n)*n*dx for n, dx in zip(co.shape, cell)]
    r = np.sqrt(sum(q*q for q in np.meshgrid(*offsets, indexing='ij', sparse=True)))
    taper = np.ones(co.shape)
    if r_max is not None:
        taper = (r <= r_max).astype(float)
        if window == 'cosine':
            taper *= 0.5*(1+np.cos(np.pi*r/r_max))
    weighted_xi = xi*taper
    vk = float(np.prod(cell))
    good = p['modes'] > 0
    radial = np.full_like(p['k'], np.nan)
    for i in np.flatnonzero(good):
        radial[i] = vk*np.sum(weighted_xi*spherical_jn(0, p['k'][i]*r))
    # Same real-space taper, but discrete Fourier lattice directions/bins.
    windowed_modes = vk*np.fft.fftn(weighted_xi).real
    axes = [2*np.pi*np.fft.fftfreq(n, dx) for n, dx in zip(co.shape, cell)]
    kmag = np.sqrt(sum(q*q for q in np.meshgrid(*axes, indexing='ij', sparse=True)))
    bins = np.searchsorted(k_edges, kmag, side='right')-1
    valid = (kmag > 0)&(bins >= 0)&(bins < len(p['k']))
    summed = np.bincount(bins[valid], weights=windowed_modes[valid], minlength=len(p['k']))
    windowed = np.full_like(p['k'], np.nan)
    np.divide(summed, p['modes'], out=windowed, where=good)
    pair = np.full_like(p['k'], np.nan)
    pairs_used = 0
    if run_pairs and good.any():
        geometry = Geometry(tuple(box))
        apos, a, av = grid_to_samples(co, geometry)
        bpos, b, bv = grid_to_samples(delta_f, geometry)
        result = estimate_cross_power(apos, a, bpos, b, p['k'][good],
            geometry=geometry, r_max=r_max, window=window,
            co_sample_volumes=av, lya_sample_volumes=bv, max_pairs=max_pairs)
        pair[good] = result.power[0]
        pairs_used = result.pairs_used
    return dict(k=p['k'], k_edges=p['k_edges'], modes=p['modes'],
                fft_cross=p['P_CO_F'], fft_windowed=windowed,
                radial_reference=radial, pair_cross=pair,
                P_CO=p['P_CO'], P_F=p['P_F'], boxsize_cmpch=box,
                pairs_used=pairs_used)
