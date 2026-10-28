import sys
import types
import numpy as np
import pytest
from lya_co.lya.astrid import flux_to_delta, delta_to_flux, read_sightlines
from lya_co.spectra.multipoles import fft_multipoles, power_1d, gaussian_truth_sn
from lya_co.spectra.covariance import ensemble_sn
from lya_co.reconstruction.dachshund import reconstruct
from lya_co.estimator.sightline_pairs import sightline_pair_multipoles
from lya_co.estimator.pair_cross import Geometry, grid_to_samples, estimate_cross_power
from lya_co.estimator.power import matched_power


def test_flux_roundtrip_and_1d_parseval():
    rng = np.random.default_rng(2)
    flux = rng.random((5, 12))
    delta = flux_to_delta(flux, .7)
    np.testing.assert_allclose(delta_to_flux(delta, .7), flux)
    k, p = power_1d(delta, 40.)
    weight = np.full(len(k), 2.); weight[[0, -1]] = 1
    assert np.sum(p*weight)/40 == pytest.approx(np.var(delta))


def test_fft_multipoles_against_monopole_and_axis_mode():
    z = np.arange(8)[None, None, :]
    a = np.broadcast_to(np.cos(2*np.pi*z/8), (8,)*3)
    edges = np.array([0, 1., 2.])
    result = fft_multipoles(a, -a, 8., edges)
    old = matched_power(a, -a, 8., edges)
    np.testing.assert_allclose(result['power'][0], old['P_CO_F'], atol=1e-12)
    assert result['power'][1, 0] == pytest.approx(5*result['power'][0, 0])
    assert result['power'][2, 0] == pytest.approx(9*result['power'][0, 0])


@pytest.mark.parametrize('periodic', [True, False])
def test_binned_pairs_match_explicit_pairs(periodic):
    rng = np.random.default_rng(5)
    co = rng.normal(size=(3,)*3)
    xyz = rng.uniform(0, 9, (11, 3))
    b = rng.normal(size=11)
    k = np.array([.1, .4, .8])
    geo = Geometry((9.,)*3, periodic=periodic)
    pos, a, volumes = grid_to_samples(co, geo)
    exact = estimate_cross_power(pos, a, xyz, b, k, geometry=geo,
                                 r_max=4., window='cosine', multipoles=(0, 2, 4))
    hist = sightline_pair_multipoles(co, xyz, b, 9., k, rmax=4., dr=.0001,
                                     co_samples=co.size, periodic=periodic)
    np.testing.assert_allclose(hist['power'], exact.power, atol=.002, rtol=.001)


def test_ensemble_sn_and_small_ensemble_guard():
    rng = np.random.default_rng(7)
    samples = rng.normal(size=(5000, 2, 3))
    result = ensemble_sn(np.ones((2, 3)), samples)
    assert result['sn_total'] == pytest.approx(np.sqrt(6), rel=.06)
    with pytest.raises(ValueError):
        ensemble_sn(np.ones((2, 3)), samples[:8])
    with pytest.raises(np.linalg.LinAlgError):
        ensemble_sn(np.ones((2, 3)), np.ones((30, 2, 3)))


def test_reader_uses_axis_positions_and_reproducible_noise(monkeypatch, tmp_path):
    class FakeSpectra:
        def __init__(self, **kwargs):
            self.box = 12000.
            self.axis = np.array([1, 3, 3, 3])
            self.cofm = np.array([[0, 0, 0], [1000, 2000, 0], [3000, 4000, 0], [5000, 6000, 0]])
        def get_tau(self, **kwargs):
            return np.ones((4, 12))
    module = types.ModuleType('fake_spectra.plot_spectra')
    module.PlottingSpectra = FakeSpectra
    monkeypatch.setitem(sys.modules, 'fake_spectra', types.ModuleType('fake_spectra'))
    monkeypatch.setitem(sys.modules, 'fake_spectra.plot_spectra', module)
    a = read_sightlines(tmp_path/'test.h5', 2, 42, np.exp(-1), stride=3)
    assert np.all(a['ids'] != 0)
    np.testing.assert_allclose(a['delta'], 0)
    np.testing.assert_array_equal(a['positions'][:4, 2], [0, 3, 6, 9])
    b = read_sightlines(tmp_path/'test.h5', 2, 42, np.exp(-1), stride=3, noise_seed=1)
    np.testing.assert_array_equal(a['positions'], b['positions'])
    assert np.any(b['delta'] != a['delta'])


def test_dachshund_file_contract_with_stub_executable(tmp_path):
    # Tests only process/file orchestration, not Dachshund reconstruction accuracy.
    exe = tmp_path/'stub.py'
    exe.write_text('#!'+sys.executable+'\n'
                   'import numpy as np\n'
                   'cfg=dict(line.strip().split(" = ") for line in open("input.cfg"))\n'
                   'pixels=np.fromfile(cfg["pixel_data_path"],dtype="f8").reshape(-1,5)\n'
                   'assert pixels.shape==(2,5)\n'
                   'assert np.all(pixels[:,3]==0.02)\n'
                   'np.arange(int(cfg["map_nx"])**3,dtype="f8").tofile(cfg["map_path"])\n')
    exe.chmod(0o755)
    result = reconstruct(exe, np.array([[1, 2, 3], [4, 5, 6]]), np.array([-.2, .3]),
                         .02, 10., 3, tmp_path/'work')
    assert result.shape == (3,)*3
    assert result[0, 0, 1] == 1
    assert result[1, 0, 0] == 9


def test_gaussian_truth_sn_finite():
    rng = np.random.default_rng(8)
    a = rng.normal(size=(8,)*3)
    result = gaussian_truth_sn(a, -.5*a+rng.normal(size=a.shape), 20., np.linspace(0, 1., 5))
    assert np.isfinite(result['sn_total']) and result['sn_total'] > 0
    assert np.all(result['sigma'][np.isfinite(result['sigma'])] > 0)


def test_dachshund_rejects_nonconvergence(tmp_path):
    exe = tmp_path/'failed_solver.py'
    exe.write_text('#!'+sys.executable+'\nprint("[WARNING] PCG reached max iteration before stop condition.")\n')
    exe.chmod(0o755)
    with pytest.raises(RuntimeError, match='did not converge'):
        reconstruct(exe, np.array([[1., 2., 3.]]), np.array([.1]),
                    .02, 10., 3, tmp_path/'failed')
