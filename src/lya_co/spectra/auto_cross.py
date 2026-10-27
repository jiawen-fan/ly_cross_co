"""Full-map spectra and optional distinct-sightline Ly-alpha pair auto power."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_legendre, spherical_jn
from .multipoles import power_1d


def sparse_auto_multipoles(sightlines, k, rmax=80., dr=.1, anchors=20000,
                           seed=42, ells=(0, 2, 4)):
    """Cross different sightlines to exclude same-skewer alias and pixel noise.

    Uniform random sightlines, equal LOS pixel spacing, periodic box assumed.
    Uniformly sampled anchor pixels approximate the ordered pair integral.
    Normalization V/[Nanchor*(Nsightline-1)*Npixel_per_sightline].
    No per-skewer mean subtraction: one sample-wide delta_F mean is removed.
    """
    values = np.asarray(sightlines['values'], dtype=float)
    if values.ndim != 2 or len(values) < 2 or rmax <= 0 or dr <= 0 or anchors < 1:
        raise ValueError('Require at least two sightlines and positive pair settings')
    ns, nz = values.shape
    pos = np.asarray(sightlines['positions'], dtype=float)
    box = float(sightlines['box'])
    if pos.shape != (values.size, 3) or not np.all(np.isfinite(values)):
        raise ValueError('Positions must match sightline-major values')
    # read_sightlines stores a contiguous LOS for each sightline.
    if not np.allclose(pos.reshape(ns, nz, 3)[:, :, :2], pos.reshape(ns, nz, 3)[:, :1, :2]):
        raise ValueError('Expected sightline-major ordering with constant x,y per LOS')
    ids = np.sort(np.random.default_rng(seed).choice(values.size, min(anchors, values.size), replace=False))
    delta = values.ravel()-values.mean()
    tree = cKDTree(pos, boxsize=box)
    edges = np.linspace(0, rmax, int(np.ceil(rmax/dr))+1)
    radius = (edges[:-1]+edges[1:])/2
    hist = np.zeros((len(ells), len(radius)))
    zero = 0.
    npairs = 0
    for i in ids:
        js = np.asarray(tree.query_ball_point(pos[i], rmax), dtype=int)
        js = js[js//nz != i//nz]  # exclude all same-sightline pairs, not just self pairs
        if not len(js):
            continue
        dp = pos[i]-pos[js]
        dp -= box*np.rint(dp/box)
        radius_ij = np.linalg.norm(dp, axis=1)
        w = delta[i]*delta[js]
        zero += w[radius_ij == 0].sum()
        good = radius_ij > 0
        r, dp, w = radius_ij[good], dp[good], w[good]
        w *= .5*(1+np.cos(np.pi*r/rmax))
        mu = dp[:, 2]/r
        npairs += len(js)
        for j, ell in enumerate(ells):
            hist[j] += np.histogram(r, edges, weights=w*eval_legendre(ell, mu))[0]
    norm = box**3/(len(ids)*(ns-1)*nz)
    k = np.asarray(k)
    good = np.isfinite(k)
    power = np.full((len(ells), len(k)), np.nan)
    for j, ell in enumerate(ells):
        power[j, good] = norm*(2*ell+1)*(-1)**(ell//2)*(spherical_jn(ell, k[good, None]*radius)@hist[j])
        if ell == 0:
            power[j, good] += norm*zero
    return dict(power=power, anchor_ids=ids, pairs_used=npairs)


def sampling_noise_monopole(values, box, k, nmu=64):
    """Uniform-sightline approximation dperp^2 P1D averaged over |mu|.

    Similar to LiLa's sampling-noise model, but assumes equal weights (q=1).
    Includes pixel noise if present in values. Does not subtract this term from
    the distinct-sightline estimator, which already excludes same-LOS pairs.
    """
    k1d, p1d = power_1d(values, box)
    mu = (np.arange(nmu)+.5)/nmu
    parallel = np.asarray(k)[:, None]*mu
    noise = box**2/len(values)*np.interp(parallel, k1d, p1d).mean(axis=1)
    noise[np.asarray(k)>k1d[-1]] = np.nan
    return noise


def save_auto_cross(output, k, co_auto, lya_autos, crosses, extra=None):
    """Write like-for-like axes/units, retaining signed multipoles."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    arrays = {'k': k, 'ells': np.array([0, 2, 4]), 'co_auto': co_auto}
    arrays.update({'lya_auto_'+name: value for name, value in lya_autos.items()})
    arrays.update({'cross_'+name: value for name, value in crosses.items()})
    if extra:
        arrays.update(extra)
    np.savez(output/'auto_cross_spectra.npz', **arrays)
    columns, labels = [k], ['k']
    for name, value in arrays.items():
        if name.startswith(('co_auto', 'lya_auto_', 'cross_')):
            for j, ell in enumerate((0, 2, 4)):
                columns.append(value[j]); labels.append(f'{name}_ell{ell}')
    np.savetxt(output/'auto_cross_spectra.csv', np.column_stack(columns),
               delimiter=',', header=','.join(labels), comments='')
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    groups = [{'CO': co_auto}, lya_autos, crosses]
    units = ['microK^2 (cMpc/h)^3', '(cMpc/h)^3', 'microK (cMpc/h)^3']
    names = ['CO auto', 'Ly-alpha auto', 'CO x Ly-alpha']
    for row, group in enumerate(groups):
        for j, ell in enumerate((0, 2, 4)):
            ax = axes[row, j]
            for name, power in group.items():
                ax.plot(k, power[j], label=name)
            ax.axhline(0, color='gray', lw=.5)
            ax.set(xlabel='k [h/cMpc]', ylabel=units[row], title=f'{names[row]}, ell={ell}')
            if j == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'auto_cross_multipoles.png', dpi=160)
    plt.close(fig)
