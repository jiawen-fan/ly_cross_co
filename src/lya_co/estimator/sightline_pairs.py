"""Radially binned pair multipoles at actual sightline positions.

CO cells may be uniformly subsampled for Monte Carlo volume integration.
This is a pair-count estimator, not a zero-filled FFT or survey deconvolution.
"""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_legendre, spherical_jn


def sightline_pair_multipoles(co, positions, delta, box, k, rmax=80., dr=.1,
                             co_samples=20000, seed=42, ells=(0, 2, 4),
                             window='cosine', periodic=True):
    if rmax <= 0 or dr <= 0 or box <= 0 or window not in ('cosine', 'tophat'):
        raise ValueError('Invalid geometry or radial window')
    positions, delta, k = np.asarray(positions), np.asarray(delta), np.asarray(k)
    if positions.shape != (len(delta), 3) or not len(delta) or not np.all(np.isfinite(delta)):
        raise ValueError('Invalid sightline samples')
    if np.any(positions < 0) or np.any(positions >= box):
        raise ValueError('Positions must be in [0,box)')
    if co_samples <= 0:
        raise ValueError('co_samples must be positive; use co.size for all voxels')
    count = min(co_samples, co.size)
    ids = np.sort(np.random.default_rng(seed).choice(co.size, count, replace=False))
    centers = (np.array(np.unravel_index(ids, co.shape)).T+.5)*box/np.array(co.shape)
    # Global CO mean, selected-sightline Ly-alpha mean: same convention as FFT maps.
    av = co.ravel()[ids]-co.mean(dtype=float)
    bv = delta-delta.mean(dtype=float)
    tree = cKDTree(positions, boxsize=box if periodic else None)
    radial_edges = np.linspace(0, rmax, int(np.ceil(rmax/dr))+1)
    radius = .5*(radial_edges[1:]+radial_edges[:-1])
    hist = np.zeros((len(ells), len(radius)))
    zero = 0.
    pairs = 0
    # Per-point neighbor lists keep memory bounded on large ASTRID catalogs.
    for pos, value in zip(centers, av):
        js = tree.query_ball_point(pos, rmax)
        if not js:
            continue
        dp = pos-positions[js]
        if periodic:
            dp -= box*np.rint(dp/box)
        r = np.linalg.norm(dp, axis=1)
        weights = value*bv[js]
        pairs += len(r)
        zero += weights[r == 0].sum()
        nonzero = r > 0
        r, dp, weights = r[nonzero], dp[nonzero], weights[nonzero]
        mu = dp[:, 2]/r
        if window == 'cosine':
            weights *= .5*(1+np.cos(np.pi*r/rmax))
        for j, ell in enumerate(ells):
            hist[j] += np.histogram(r, radial_edges, weights=weights*eval_legendre(ell, mu))[0]
    power = np.full((len(ells), len(k)), np.nan)
    good = np.isfinite(k)
    # (V/N_CO)*(V/N_pixels)/V. Uniform randomly selected sightlines assumed.
    normalization = box**3/(count*len(delta))
    for j, ell in enumerate(ells):
        power[j, good] = normalization*(2*ell+1)*(-1)**(ell//2) * (
            spherical_jn(ell, k[good, None]*radius) @ hist[j])
        if ell == 0:
            power[j, good] += normalization*zero
    return dict(power=power, radial_edges=radial_edges, hist=hist,
                co_ids=ids, pairs_used=pairs)
