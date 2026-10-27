"""Small-catalog, configuration-space CO x Ly-alpha cross-power prototype.

Not a drop-in HIPSTER implementation: this sums weighted field-sample pairs
and does not estimate the full survey correction from random catalogues.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.special import eval_legendre, spherical_jn


@dataclass(frozen=True)
class Geometry:
    """Cartesian measurement volume in (cMpc/h); LOS is a Cartesian axis."""

    boxsize: tuple[float, float, float]
    periodic: bool = True
    los_axis: int = 2
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        box = np.asarray(self.boxsize, dtype=float)
        origin = np.asarray(self.origin, dtype=float)
        if box.shape != (3,) or not np.all(np.isfinite(box)) or np.any(box <= 0):
            raise ValueError("boxsize must contain three positive, finite lengths")
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("origin must contain three finite coordinates")
        if self.los_axis not in (0, 1, 2):
            raise ValueError("los_axis must be 0, 1, or 2")


@dataclass(frozen=True)
class PairSpectrum:
    """Power[ell_index, k_index], with units from input fields times volume."""

    k: np.ndarray
    multipoles: tuple[int, ...]
    power: np.ndarray
    pairs_used: int
    volume: float


def _samples(positions, values, volumes, geometry, label):
    xyz = np.asarray(positions, dtype=float)
    value = np.asarray(values, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz):
        raise ValueError(f"{label} positions must be a nonempty (N, 3) array")
    if value.shape != (len(xyz),) or not np.all(np.isfinite(value)):
        raise ValueError(f"{label} values must be finite with shape (N,)")
    box = np.asarray(geometry.boxsize)
    origin = np.asarray(geometry.origin)
    if not np.all(np.isfinite(xyz)) or np.any(xyz < origin) or np.any(xyz >= origin + box):
        raise ValueError(f"{label} positions must lie inside the geometry box")
    if volumes is None:
        weights = np.full(len(xyz), float(np.prod(box)) / len(xyz))
    else:
        weights = np.asarray(volumes, dtype=float)
    if weights.shape != (len(xyz),) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"{label} sample volumes must be positive, finite (N,) values")
    return xyz, value, weights


def grid_to_samples(field, geometry: Geometry, mask=None, selection_probability=None):
    """Convert a Cartesian cube to (positions, values, integration volumes).

    ``mask`` retains measured voxels only. If samples are selected randomly
    with a known probability p, ``selection_probability=p`` applies 1/p
    weighting; this does not correct structured sightline aliasing or noise.
    """
    data = np.asarray(field, dtype=float)
    if data.ndim != 3 or any(n == 0 for n in data.shape):
        raise ValueError("field must be a nonempty 3D grid")
    if mask is None:
        select = np.ones(data.shape, dtype=bool)
    else:
        select = np.asarray(mask)
        if select.shape != data.shape or select.dtype != np.dtype(bool):
            raise ValueError("mask must be a boolean array matching field shape")
    if not np.any(select) or not np.all(np.isfinite(data[select])):
        raise ValueError("selected field values must be nonempty and finite")
    if selection_probability is None:
        probability = np.ones(np.count_nonzero(select), dtype=float)
    else:
        prob = np.asarray(selection_probability, dtype=float)
        if prob.ndim == 0:
            probability = np.full(np.count_nonzero(select), prob.item())
        elif prob.shape == data.shape:
            probability = prob[select]
        else:
            raise ValueError("selection_probability must be scalar or grid-shaped")
        if np.any(probability <= 0) or np.any(probability > 1) or not np.all(np.isfinite(probability)):
            raise ValueError("selection probabilities must be finite and in (0, 1]")
    cell = np.asarray(geometry.boxsize, dtype=float) / np.asarray(data.shape)
    centers = np.asarray(geometry.origin) + (np.argwhere(select) + 0.5) * cell
    volumes = np.full(len(centers), float(np.prod(cell))) / probability
    return centers, data[select], volumes


def estimate_cross_power(
    co_positions,
    co_delta,
    lya_positions,
    lya_delta,
    k,
    *,
    geometry: Geometry,
    r_max: float | None = None,
    window: str | Callable[[np.ndarray], np.ndarray] = "tophat",
    multipoles: tuple[int, ...] = (0,),
    co_sample_volumes=None,
    lya_sample_volumes=None,
    geometry_correction: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    subtract_mean: bool = True,
    chunk_size: int = 128,
    max_pairs: int = 2_000_000,
) -> PairSpectrum:
    r"""Pair-sum estimate of the cross-power monopole/even multipoles.

    P_l(k) = (2l+1)(-1)^(l/2) / V * sum_ij [w_i w_j a_i b_j
                 W(r_ij) C(r_ij, mu_ij) j_l(k*r_ij) L_l(mu_ij)].

    The units are (units of CO) * (units of Ly-alpha) * (cMpc/h)^3.
    ``co_sample_volumes`` and ``lya_sample_volumes`` are quadrature volumes,
    *not* inverse-noise weights. Default V/N assumes uniform sampling.
    Masked or sparse survey sampling is not automatically deconvolved.
    ``geometry_correction`` is an optional externally calibrated multiplier
    C(r, mu), not an internally estimated HIPSTER survey correction.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError("geometry must be a Geometry instance")
    ks = np.asarray(k, dtype=float)
    if ks.ndim != 1 or not len(ks) or not np.all(np.isfinite(ks)) or np.any(ks < 0):
        raise ValueError("k must be a nonempty array of finite nonnegative wavenumbers")
    if not isinstance(multipoles, (list, tuple)) or not multipoles or any(
        not isinstance(ell, (int, np.integer)) or ell < 0 or ell % 2
        for ell in multipoles
    ) or len(set(multipoles)) != len(multipoles):
        raise ValueError("multipoles must be unique, nonnegative even integers")
    if not isinstance(chunk_size, int) or not isinstance(max_pairs, int) or chunk_size <= 0 or max_pairs <= 0:
        raise ValueError("chunk_size and max_pairs must be positive")
    if r_max is not None and (not np.isfinite(r_max) or r_max <= 0):
        raise ValueError("r_max must be finite and positive")
    if isinstance(window, str):
        if window not in ("tophat", "cosine"):
            raise ValueError("window must be 'tophat', 'cosine', or a callable")
        if window == "cosine" and r_max is None:
            raise ValueError("cosine window requires r_max")
    elif not callable(window):
        raise TypeError("window must be a string or callable")
    if geometry_correction is not None and not callable(geometry_correction):
        raise TypeError("geometry_correction must be callable")

    a_pos, a, a_vol = _samples(co_positions, co_delta, co_sample_volumes, geometry, "CO")
    b_pos, b, b_vol = _samples(lya_positions, lya_delta, lya_sample_volumes, geometry, "Ly-alpha")
    if len(a) * len(b) > max_pairs:
        raise ValueError("Too many pairs for this reference implementation; use a KD-tree/Corrfunc backend")
    if subtract_mean:
        a = a - np.average(a, weights=a_vol)
        b = b - np.average(b, weights=b_vol)

    box = np.asarray(geometry.boxsize)
    volume = float(np.prod(box))
    result = np.zeros((len(multipoles), len(ks)), dtype=float)
    used = 0
    for i in range(0, len(a), chunk_size):
        for j in range(0, len(b), chunk_size):
            dp = a_pos[i:i+chunk_size, None, :] - b_pos[None, j:j+chunk_size, :]
            if geometry.periodic:
                dp = dp - box * np.rint(dp / box)
            r = np.linalg.norm(dp, axis=-1)
            valid = r <= r_max if r_max is not None else np.ones(r.shape, dtype=bool)
            if not np.any(valid):
                continue
            used += int(np.count_nonzero(valid))
            sep = r[valid]
            mu = np.divide(dp[..., geometry.los_axis][valid], sep,
                           out=np.zeros_like(sep), where=sep > 0)
            pair_weight = (a[i:i+chunk_size] * a_vol[i:i+chunk_size])[:, None] * (
                b[j:j+chunk_size] * b_vol[j:j+chunk_size])[None, :]
            pair_weight = pair_weight[valid]
            if isinstance(window, str):
                taper = np.ones_like(sep) if window == "tophat" else 0.5 * (1 + np.cos(np.pi * sep / r_max))
            else:
                taper = np.broadcast_to(np.asarray(window(sep), dtype=float), sep.shape)
            if not np.all(np.isfinite(taper)):
                raise ValueError("window returned nonfinite values")
            pair_weight = pair_weight * taper
            if geometry_correction is not None:
                corr = np.broadcast_to(np.asarray(geometry_correction(sep, mu), dtype=float), sep.shape)
                if not np.all(np.isfinite(corr)) or np.any(corr <= 0):
                    raise ValueError("geometry_correction must return positive finite values")
                pair_weight = pair_weight * corr
            for idx, ell in enumerate(multipoles):
                angular = eval_legendre(ell, mu)
                prefactor = (2*ell + 1) * (-1)**(ell // 2) / volume
                for ik, kval in enumerate(ks):
                    result[idx, ik] += prefactor * np.dot(pair_weight * angular,
                                                          spherical_jn(ell, kval * sep))
    return PairSpectrum(k=ks.copy(), multipoles=tuple(multipoles),
                        power=result, pairs_used=used, volume=volume)

