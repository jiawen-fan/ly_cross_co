"""Empirical covariance and S/N for matched independent mock measurements."""
import numpy as np


def ensemble_sn(signal, realizations):
    """Return per-bin and joint S/N; no diagonal or singular inverse shortcut.

    Flatten order is ell then k. Require Nmock > Ndata+2 and a positive definite
    covariance. Hartlap precision correction assumes independent Gaussian mocks.
    The caller must state whether mocks include sample variance or only noise.
    """
    signal = np.asarray(signal, dtype=float)
    samples = np.asarray(realizations, dtype=float)
    if samples.shape[1:] != signal.shape:
        raise ValueError('Ensemble dimensions must match signal')
    flat = samples.reshape(len(samples), -1)
    valid = np.isfinite(signal.ravel()) & np.all(np.isfinite(flat), axis=0)
    s, x = signal.ravel()[valid], flat[:, valid]
    n, p = x.shape
    if p == 0 or n <= p+2:
        raise ValueError(f'Need more than Ndata+2 independent mocks: Nmock={n}, Ndata={p}')
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    np.linalg.cholesky(covariance)  # reject singular/indefinite matrices
    hartlap = (n-p-2)/(n-1)
    precision_signal = np.linalg.solve(covariance, s)*hartlap
    error = np.full(signal.size, np.nan)
    error[valid] = np.sqrt(np.diag(covariance))
    per_bin = np.full(signal.size, np.nan)
    per_bin[valid] = np.abs(s)/error[valid]
    return dict(covariance=covariance, valid=valid.reshape(signal.shape),
                sigma=error.reshape(signal.shape), sn=per_bin.reshape(signal.shape),
                sn_total=float(np.sqrt(max(0., s@precision_signal))),
                hartlap=hartlap, nmock=n)
