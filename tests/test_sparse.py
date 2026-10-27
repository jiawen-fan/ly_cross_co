import numpy as np
from lya_co.estimator.compare import compare_estimators
from lya_co.estimator.pair_cross import Geometry,grid_to_samples,estimate_cross_power

def test_sparse_compression_matches_actual_selected_pairs():
    rng=np.random.default_rng(8)
    co,lya=rng.normal(size=(2,4,4,4))
    mask=np.zeros((4,4,4),bool);mask[0,1,:]=True;mask[3,2,:]=True
    sparse=np.zeros_like(lya);sparse[mask]=(lya[mask]-lya[mask].mean())*8
    r=compare_estimators(co,sparse,10,[0,.7,1.5],r_max=4,window='cosine',run_pairs=False)
    geo=Geometry((10.,)*3)
    ap,av,aw=grid_to_samples(co,geo)
    bp,bv,bw=grid_to_samples(lya,geo,mask,selection_probability=1/8)
    expected=estimate_cross_power(ap,av,bp,bv,r['k'],geometry=geo,r_max=4,window='cosine',co_sample_volumes=aw,lya_sample_volumes=bw)
    np.testing.assert_allclose(r['radial_reference'],expected.power[0],atol=1e-11)
