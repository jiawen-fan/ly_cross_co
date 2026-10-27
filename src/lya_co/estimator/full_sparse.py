"""Full periodic map baselines and grid-aligned sparse sightline comparison."""
import numpy as np
from .compare import compare_estimators


def sparse_sightlines(delta_f, count, seed=42):
    """Select unique z columns and weight their mean-subtracted values by 1/p."""
    nx, ny, _ = delta_f.shape
    if not 1 <= count <= nx * ny:
        raise ValueError('Sightline count must be between 1 and nx*ny')
    ids = np.sort(np.random.default_rng(seed).choice(nx * ny, count, replace=False))
    x, y = np.unravel_index(ids, (nx, ny))
    values = np.asarray(delta_f[x, y], dtype=float)
    mean = float(values.mean())
    probability = count / (nx * ny)
    accumulator = np.zeros(delta_f.shape, dtype=float)
    accumulator[x, y] = (values - mean) / probability
    return accumulator, x, y, mean, probability


def compare_full_sparse(co, delta_f, boxsize, k_edges, sightlines=1000,
                        seed=42, r_max=80., window='cosine'):
    """Compare full FFT shells with sparse pairs, retaining matched-window controls.

    Sparse pairs use only selected Ly-alpha columns. Full-map auto spectra and
    r refer exclusively to complete maps; no sparse-pair covariance is assumed.
    Normalization: FFT V/Ncell**2; pair volumes dV_CO * dV_F/p divided by V.
    """
    if co.shape != delta_f.shape:
        raise ValueError('CO and Ly-alpha must have the same aligned grid')
    sparse, x, y, mean, probability = sparse_sightlines(delta_f, sightlines, seed)
    options = dict(r_max=r_max, window=window, run_pairs=False)
    full = compare_estimators(co, delta_f, boxsize, k_edges, **options)
    selected = compare_estimators(co, sparse, boxsize, k_edges, **options)
    denominator = np.sqrt(full['P_CO'] * full['P_F'])
    correlation = np.full_like(full['k'], np.nan)
    np.divide(full['fft_cross'], denominator, out=correlation, where=denominator > 0)
    return dict(k=full['k'], k_edges=full['k_edges'], modes=full['modes'],
                full_fft=full['fft_cross'], full_windowed_fft=full['fft_windowed'],
                full_pair_monopole=full['radial_reference'],
                sparse_pair_monopole=selected['radial_reference'],
                sparse_windowed_fft=selected['fft_windowed'],
                P_CO=full['P_CO'], P_F=full['P_F'], r_full=correlation,
                skewer_ix=x, skewer_iy=y, sample_mean_delta_f=mean,
                full_mean_delta_f=float(np.mean(delta_f, dtype=float)),
                selection_probability=probability, boxsize_cmpch=full['boxsize_cmpch'])
