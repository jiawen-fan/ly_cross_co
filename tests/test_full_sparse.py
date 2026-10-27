import importlib.util
from pathlib import Path
import h5py
import numpy as np
from lya_co.estimator.full_sparse import compare_full_sparse, sparse_sightlines
from lya_co.estimator.power import matched_power


def test_all_sightlines_and_full_baseline():
    rng = np.random.default_rng(8)
    co = rng.normal(size=(6, 6, 6))
    lya = -.3*co + rng.normal(size=co.shape)
    edges = np.linspace(0, 2, 8)
    result = compare_full_sparse(co, lya, 20., edges, 36, r_max=8.)
    expected = matched_power(co, lya, 20., edges)
    np.testing.assert_allclose(result['full_fft'], expected['P_CO_F'])
    np.testing.assert_allclose(result['sparse_pair_monopole'], result['full_pair_monopole'], atol=1e-12)
    np.testing.assert_allclose(result['sparse_windowed_fft'], result['full_windowed_fft'], atol=1e-12)
    assert np.nanmax(np.abs(result['r_full'])) <= 1 + 1e-12


def test_unselected_pixels_only_change_full_baseline():
    rng = np.random.default_rng(9)
    co = rng.normal(size=(6, 6, 6))
    lya = rng.normal(size=co.shape)
    _, x, y, _, _ = sparse_sightlines(lya, 5, 42)
    mask = np.ones(co.shape, dtype=bool)
    mask[x, y] = False
    changed = lya.copy()
    changed[mask] += 10 * co[mask]
    options = dict(boxsize=20., k_edges=np.linspace(0, 2, 8), sightlines=5, r_max=8.)
    a = compare_full_sparse(co, lya, **options)
    b = compare_full_sparse(co, changed, **options)
    np.testing.assert_allclose(a['sparse_pair_monopole'], b['sparse_pair_monopole'])
    assert not np.allclose(a['full_fft'], b['full_fft'], equal_nan=True)
    np.testing.assert_array_equal(a['modes'], b['modes'])


def test_loader_with_and_without_rebin(tmp_path):
    spec = importlib.util.spec_from_file_location('sparse_script', Path(__file__).parents[1] / 'scripts/sparse_pairs.py')
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    values = np.arange(216.).reshape(6, 6, 6)
    np.savez(tmp_path / 'co.npz', map_cube=values)
    with h5py.File(tmp_path / 'lya.h5', 'w') as f:
        f['map'] = values
    for n in [6, 3]:
        co, lya = script.load_maps(tmp_path / 'co.npz', tmp_path / 'lya.h5', n)
        assert isinstance(lya, np.ndarray)
        np.testing.assert_array_equal(co, lya)
        assert co.shape == (n,)*3
        assert co.mean() == values.mean()
