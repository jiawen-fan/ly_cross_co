"""Small analytic tests: they do not require ASTRID datasets or HPC."""
import numpy as np
import pytest

from lya_co.estimator.pair_cross import Geometry, estimate_cross_power, grid_to_samples


BOX = Geometry(boxsize=(10.0, 10.0, 10.0))


def test_constant_fields_center_to_zero():
    xyz = np.array([[1., 1., 1.], [2., 2., 2.], [3., 4., 5.]])
    p = estimate_cross_power(xyz, [4, 4, 4], xyz, [5, 5, 5],
                             [0., 0.1], geometry=BOX, multipoles=(0, 2))
    np.testing.assert_allclose(p.power, 0., atol=1e-12)


def test_dc_normalization_with_explicit_weights():
    a = np.array([[1., 1., 1.]])
    b = np.array([[2., 1., 1.]])
    p = estimate_cross_power(a, [2.], b, [3.], [0.], geometry=BOX,
                             subtract_mean=False, co_sample_volumes=[10.],
                             lya_sample_volumes=[20.])
    assert p.power[0, 0] == pytest.approx(2. * 3. * 10. * 20. / 1000.)


def test_sign_and_swap_symmetry():
    co_pos = np.array([[1., 2., 3.], [8., 8., 9.], [4., 6., 2.]])
    f_pos = np.array([[2., 1., 5.], [9., 8., 1.]])
    co = np.array([1., 2., -3.])
    f = np.array([2., -5.])
    params = dict(k=[0.1, 0.4], geometry=BOX, multipoles=(0, 2),
                  subtract_mean=False, r_max=4.5, window="cosine")
    p = estimate_cross_power(co_pos, co, f_pos, f, **params)
    q = estimate_cross_power(f_pos, f, co_pos, co, **params)
    n = estimate_cross_power(co_pos, -co, f_pos, f, **params)
    np.testing.assert_allclose(p.power, q.power, atol=1e-12)
    np.testing.assert_allclose(n.power, -p.power, atol=1e-12)


def test_periodic_wrap_and_open_geometry():
    co = np.array([[0.5, 1., 1.]])
    f = np.array([[9.5, 1., 1.]])
    args = dict(co_positions=co, co_delta=[1.], lya_positions=f,
                lya_delta=[1.], k=[0.], r_max=2., subtract_mean=False)
    periodic = estimate_cross_power(**args, geometry=BOX)
    open_box = estimate_cross_power(**args,
        geometry=Geometry(boxsize=(10., 10., 10.), periodic=False))
    assert periodic.pairs_used == 1
    assert open_box.pairs_used == 0
    assert periodic.power[0, 0] == pytest.approx(1000.)
    assert open_box.power[0, 0] == pytest.approx(0.)


def test_window_and_correction_functions():
    args = dict(co_positions=[[1, 1, 1]], co_delta=[2.],
                lya_positions=[[2, 1, 1]], lya_delta=[3.], k=[0.],
                geometry=BOX, r_max=2., subtract_mean=False)
    simple = estimate_cross_power(**args)
    taper = estimate_cross_power(**args, window="cosine")
    corrected = estimate_cross_power(**args,
        geometry_correction=lambda r, mu: np.full_like(r, 2.))
    assert taper.power[0, 0] == pytest.approx(simple.power[0, 0] / 2.)
    assert corrected.power[0, 0] == pytest.approx(simple.power[0, 0] * 2.)


def test_input_errors_and_pair_limit():
    with pytest.raises(ValueError, match="boxsize"):
        Geometry(boxsize=(10., -1., 10.))
    with pytest.raises(ValueError, match="cosine"):
        estimate_cross_power([[1, 1, 1]], [1], [[2, 2, 2]], [1], [0.1],
                             geometry=BOX, window="cosine")
    with pytest.raises(ValueError, match="Too many pairs"):
        estimate_cross_power([[1, 1, 1], [2, 1, 1]], [1, 2],
                             [[2, 2, 2], [3, 3, 3]], [1, 2], [0.1],
                             geometry=BOX, max_pairs=1)


def test_grid_adapter_mask_and_sampling_probability():
    g = Geometry((4., 4., 4.))
    cube = np.arange(8.).reshape(2, 2, 2)
    mask = np.zeros_like(cube, dtype=bool)
    mask[0, 0, 0] = True
    mask[1, 1, 1] = True
    xyz, values, volumes = grid_to_samples(cube, g, mask, selection_probability=0.25)
    np.testing.assert_allclose(xyz, [[1., 1., 1.], [3., 3., 3.]])
    np.testing.assert_allclose(values, [0., 7.])
    np.testing.assert_allclose(volumes, [32., 32.])

