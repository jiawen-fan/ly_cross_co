import numpy as np
import pytest
from lya_co.estimator.compare import compare_estimators


@pytest.mark.parametrize('shape', [(4,4,4),(5,3,4)])
@pytest.mark.parametrize('window,rmax', [('tophat',None),('tophat',3.),('cosine',4.)])
def test_pair_matches_radial_reference(shape, window, rmax):
    rng = np.random.default_rng(19)
    a,b = rng.normal(size=(2,*shape))
    out = compare_estimators(a+15,b-8,[10.,12.,14.],np.linspace(0,1.4,7),
                             window=window,r_max=rmax)
    good = out['modes'] > 0
    np.testing.assert_allclose(out['pair_cross'][good],out['radial_reference'][good],rtol=1e-11,atol=1e-11)
    if rmax is None:
        np.testing.assert_allclose(out['fft_cross'][good],out['fft_windowed'][good],rtol=1e-11,atol=1e-11)


def test_pair_limit_and_skip():
    a = np.ones((4,4,4))
    with pytest.raises(ValueError,match='Pair limit'):
        compare_estimators(a,a,10,[0,1],max_pairs=2)
    out = compare_estimators(a,a,10,[0,1],run_pairs=False,max_pairs=2)
    assert np.isnan(out['pair_cross']).all()
