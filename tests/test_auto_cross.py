import h5py
import numpy as np
from scipy.special import eval_legendre, spherical_jn
from lya_co.spectra.auto_cross import sparse_auto_multipoles, sampling_noise_monopole
from lya_co.lya.astrid import load_maps


def test_distinct_sightline_auto_matches_explicit_pair_sum():
    box = 10.
    xy = np.array([[1., 2.], [3., 4.], [5., 6.]])
    z = np.array([1., 4., 7.])
    pos = np.empty((3, 3, 3))
    pos[..., :2] = xy[:, None, :]
    pos[..., 2] = z
    values = np.random.default_rng(3).normal(size=(3, 3))
    sight = dict(values=values, positions=pos.reshape(-1, 3), box=box)
    k = np.array([.1, .5])
    result = sparse_auto_multipoles(sight, k, rmax=6., dr=.0001, anchors=9)
    delta = values.ravel()-values.mean()
    p = sight['positions']
    exact = np.zeros((3, 2))
    for i in range(9):
        for j in range(9):
            if i//3 == j//3:
                continue
            dp = p[i]-p[j]
            dp -= box*np.rint(dp/box)
            r = np.linalg.norm(dp)
            if r > 6:
                continue
            w = delta[i]*delta[j]*.5*(1+np.cos(np.pi*r/6))
            for il, ell in enumerate([0, 2, 4]):
                exact[il] += (2*ell+1)*(-1)**(ell//2)*w*eval_legendre(ell, dp[2]/r)*spherical_jn(ell, k*r)
    exact *= box**3/(9*2*3)
    np.testing.assert_allclose(result['power'], exact, atol=.004, rtol=.002)


def test_sampling_noise_scales_with_inverse_sightline_density():
    values = np.random.default_rng(5).normal(size=(4, 16))
    k = np.array([.1, .2, np.nan])
    p = sampling_noise_monopole(values, 20., k)
    doubled = sampling_noise_monopole(np.concatenate([values, values]), 20., k)
    np.testing.assert_allclose(p, doubled*2)


def test_raw_flux_map_and_delta_map_agree(tmp_path):
    delta = np.random.default_rng(7).normal(0, .1, (4,)*3)
    np.savez(tmp_path/'co.npz', map_cube=delta)
    for name, values in [('delta', delta), ('flux', (1+delta)*.7)]:
        with h5py.File(tmp_path/f'{name}.h5', 'w') as f:
            f['map'] = values
    _, a = load_maps(tmp_path/'co.npz', tmp_path/'delta.h5', 2)
    _, b = load_maps(tmp_path/'co.npz', tmp_path/'flux.h5', 2, 'flux', .7)
    np.testing.assert_allclose(a, b, atol=1e-7)
