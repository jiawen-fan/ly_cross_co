"""Periodic Cartesian spectra; no assignment/window deconvolution.

DFT convention: P_ab(k) = V/Ncell**2 Re[FFT(a) FFT(b)*].
Positions/box are cMpc/h, k is h/cMpc. Power carries field_a*field_b
units times (cMpc/h)^3. Both signs of each Fourier mode are counted.
"""
import numpy as np


def matched_power(co, delta_f, boxsize, k_edges):
    """Measure both autos and the cross on exactly the same Fourier modes.

    Inputs must share shape, origin, axis order, voxel locations, redshift,
    and real/redshift-space convention. delta_f is already F/<F>-1.
    This estimator requires a complete periodic volume, not masked/sparse data.
    Empty bins remain NaN with zero modes. Bins are [left, right).
    """
    co, delta_f = np.asarray(co), np.asarray(delta_f)
    if co.ndim != 3 or co.shape != delta_f.shape:
        raise ValueError('CO and delta_F must be aligned 3D arrays of equal shape')
    if not np.isrealobj(co) or not np.isrealobj(delta_f):
        raise ValueError('Input maps must be real')
    if not np.all(np.isfinite(co)) or not np.all(np.isfinite(delta_f)):
        raise ValueError('Maps must be finite; do not fill missing sightlines with zeros')
    box = np.broadcast_to(np.asarray(boxsize, dtype=float), (3,)).copy()
    edges = np.asarray(k_edges, dtype=float)
    if np.any(~np.isfinite(box)) or np.any(box <= 0):
        raise ValueError('boxsize must contain positive finite lengths')
    if edges.ndim != 1 or edges.size < 2 or not np.all(np.isfinite(edges)) or edges[0] < 0 or np.any(np.diff(edges) <= 0):
        raise ValueError('k_edges must be finite, nonnegative and strictly increasing')
    # rFFT stores only nonnegative kz. Interior kz planes represent both signs.
    fa = np.fft.rfftn(co - co.mean(dtype=np.float64))
    fb = np.fft.rfftn(delta_f - delta_f.mean(dtype=np.float64))
    nx, ny, nz = co.shape
    kx = 2*np.pi*np.fft.fftfreq(nx, box[0]/nx)
    ky = 2*np.pi*np.fft.fftfreq(ny, box[1]/ny)
    kz = 2*np.pi*np.fft.rfftfreq(nz, box[2]/nz)
    weights = np.full(kz.size, 2, dtype=int)
    weights[0] = 1
    if nz % 2 == 0:
        weights[-1] = 1
    weights = np.broadcast_to(weights, (ny, kz.size))
    nb = edges.size - 1
    counts = np.zeros(nb, dtype=np.int64)
    sums = np.zeros((4, nb))
    norm = np.prod(box) / co.size**2
    for i, x in enumerate(kx):
        k = np.sqrt(x*x + ky[:, None]**2 + kz[None, :]**2)
        bins = np.searchsorted(edges, k, side='right') - 1
        keep = (k > 0) & (bins >= 0) & (bins < nb)
        b, w = bins[keep], weights[keep]
        counts += np.bincount(b, weights=w, minlength=nb).astype(np.int64)
        values = (k, norm*abs(fa[i])**2, norm*abs(fb[i])**2,
                  norm*(fa[i]*fb[i].conj()).real)
        for total, value in zip(sums, values):
            total += np.bincount(b, weights=w*value[keep], minlength=nb)
    means = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return dict(k=means[0], P_CO=means[1], P_F=means[2], P_CO_F=means[3],
                modes=counts, k_edges=edges, boxsize_cmpch=box)


def power_spectrum(field, boxsize, field_b=None, k_edges=None):
    """Convenience auto/cross wrapper retaining the old k/Pk/nmodes keys."""
    if k_edges is None:
        box = np.broadcast_to(np.asarray(boxsize, dtype=float), (3,))
        dk = 2*np.pi/max(box)
        limit = min(np.pi*np.asarray(field.shape)/box)
        k_edges = np.arange(0, limit, dk)
        k_edges = np.append(k_edges, limit)
    p = matched_power(field, field if field_b is None else field_b, boxsize, k_edges)
    return dict(k=p['k'], Pk=p['P_CO_F'], nmodes=p['modes'], k_edges=p['k_edges'])
