"""Common Fourier bins for even multipoles and LOS 1D power."""
import numpy as np
from scipy.special import eval_legendre


def power_1d(delta, length):
    """Last axis is uniformly spaced LOS; returns two-sided P1D (not doubled)."""
    delta = np.asarray(delta, dtype=float)
    n = delta.shape[-1]
    transform = np.fft.rfft(delta - delta.mean())
    power = length / n**2 * np.abs(transform)**2
    return 2*np.pi*np.fft.rfftfreq(n, length/n), power.reshape(-1, power.shape[-1]).mean(axis=0)


def fft_multipoles(a, b, box, edges, ells=(0, 2, 4), los_axis=2):
    """Full +/-k shell counts; P_l=(2l+1)<Re[a_k b_k*] L_l(mu)> V/N^2."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.ndim != 3 or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Require matching finite 3D maps')
    edges = np.asarray(edges)
    if box <= 0 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
        raise ValueError('Invalid box or k edges')
    fa = np.fft.fftn(a-a.mean(dtype=float))
    fb = np.fft.fftn(b-b.mean(dtype=float))
    axes = [2*np.pi*np.fft.fftfreq(n, box/n) for n in a.shape]
    mesh = np.meshgrid(*axes, indexing='ij', sparse=True)
    kmag = np.sqrt(sum(q*q for q in mesh))
    mu = np.divide(mesh[los_axis], kmag, out=np.zeros_like(kmag), where=kmag>0)
    bins = np.searchsorted(edges, kmag, side='right')-1
    nb = len(edges)-1
    valid = (kmag>0)&(bins>=0)&(bins<nb)
    ix = bins[valid]
    counts = np.bincount(ix, minlength=nb)
    def average(values):
        out = np.full(nb, np.nan)
        np.divide(np.bincount(ix, weights=values[valid], minlength=nb), counts,
                  out=out, where=counts>0)
        return out
    norm = box**3/a.size**2
    cross = (fa*fb.conj()).real*norm
    return dict(k=average(kmag), modes=counts, ells=np.array(ells),
                power=np.array([(2*l+1)*average(cross*eval_legendre(l, mu)) for l in ells]),
                auto_a=average(np.abs(fa)**2*norm), auto_b=average(np.abs(fb)**2*norm))


def gaussian_truth_sn(a, b, box, edges, nmu=8, ells=(0, 2, 4)):
    """Full-periodic-map Gaussian covariance, diagonal in k, coupled in ell.

    Estimate anisotropic auto/cross powers in |mu| cells before contracting
    Legendre weights. This is a sample-variance-only approximation, NOT valid
    for a sightline mask or Wiener reconstruction. Shot power stays in autos.
    """
    measured = fft_multipoles(a, b, box, edges, ells)
    fa = np.fft.fftn(a-a.mean(dtype=float))
    fb = np.fft.fftn(b-b.mean(dtype=float))
    axes = [2*np.pi*np.fft.fftfreq(n, box/n) for n in a.shape]
    mesh = np.meshgrid(*axes, indexing='ij', sparse=True)
    kmag = np.sqrt(sum(q*q for q in mesh))
    mu = np.divide(mesh[2], kmag, out=np.zeros_like(kmag), where=kmag>0)
    kb = np.searchsorted(edges, kmag, side='right')-1
    mb = np.minimum((np.abs(mu)*nmu).astype(int), nmu-1)
    norm = box**3/a.size**2
    aa, bb, ab = np.abs(fa)**2*norm, np.abs(fb)**2*norm, (fa*fb.conj()).real*norm
    nb = len(edges)-1
    cov = np.full((nb, len(ells), len(ells)), np.nan)
    sn2 = 0.
    for i in range(nb):
        select = (kb == i)&(kmag > 0)
        n = select.sum()
        if not n:
            continue
        block = np.zeros((len(ells), len(ells)))
        for j in range(nmu):
            cell = select & (mb == j)
            if not cell.any():
                continue
            variance = aa[cell].mean()*bb[cell].mean()+ab[cell].mean()**2
            weights = np.array([(2*l+1)*eval_legendre(l, mu[cell]) for l in ells])
            block += variance*(weights @ weights.T)/n**2
        cov[i] = block
        # Some low-k shells have too few angular directions for 3 multipoles.
        # Total S/N is restricted to the identifiable subspace; rank is saved.
        sn2 += measured['power'][:, i] @ np.linalg.pinv(block, hermitian=True) @ measured['power'][:, i]
    sigma = np.sqrt(np.diagonal(cov, axis1=1, axis2=2)).T
    return dict(covariance=cov, sigma=sigma, sn=np.abs(measured['power'])/sigma,
                sn_total=float(np.sqrt(max(0, sn2))),
                ranks=np.array([np.linalg.matrix_rank(c) if np.all(np.isfinite(c)) else 0 for c in cov]))
