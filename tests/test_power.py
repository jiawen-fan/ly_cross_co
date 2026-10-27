import unittest
import numpy as np
from lya_co.estimator.power import matched_power
from lya_co.co.co_ly_sn import combine_maps_and_sn, signal_to_noise, rebin_mean


class PowerTests(unittest.TestCase):
    def test_full_fft_and_parseval(self):
        for shape in [(6,8,10), (5,7,9)]:
            rng = np.random.default_rng(42)
            a, b = rng.normal(size=(2, *shape))
            box = np.array([30.,40.,50.])
            edges = np.linspace(0,3,19)
            p = matched_power(a,b,box,edges)
            k = np.sqrt(sum(q*q for q in np.meshgrid(*[
                2*np.pi*np.fft.fftfreq(n,L/n) for n,L in zip(shape,box)
            ], indexing='ij')))
            fa,fb = np.fft.fftn(a-a.mean()),np.fft.fftn(b-b.mean())
            cross = (fa*fb.conj()).real*np.prod(box)/a.size**2
            for i,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
                select = (k>0)&(k>=lo)&(k<hi)
                self.assertEqual(p['modes'][i],select.sum())
                if select.any():
                    np.testing.assert_allclose(p['P_CO_F'][i],cross[select].mean(),atol=1e-12)
            self.assertEqual(p['modes'].sum(),a.size-1)
            np.testing.assert_allclose(np.nansum(p['P_CO']*p['modes'])/np.prod(box),a.var())
            np.testing.assert_allclose(np.nansum(p['P_CO_F']*p['modes'])/np.prod(box),np.mean((a-a.mean())*(b-b.mean())))

    def test_sinusoid_and_negative_cross(self):
        x = np.arange(16)*2*np.pi/16
        a = np.broadcast_to(3*np.cos(x)[:,None,None],(16,16,16))
        p = matched_power(a+40,-2*a+10,16,[0,.5])
        np.testing.assert_allclose(p['P_CO']*p['modes'],3**2*16**3/2)
        np.testing.assert_allclose(p['P_CO_F'],-2*p['P_CO'])
        np.testing.assert_allclose(p['P_F'],4*p['P_CO'])
        r = combine_maps_and_sn(a,-2*a,16,[0,.5],0,0)
        np.testing.assert_allclose(r['r'],-1)
        np.testing.assert_allclose(r['sn_cross'],np.sqrt(r['modes']/2))

    def test_gaussian_noise(self):
        r = signal_to_noise([1],[4],[9],[-3],[100],1,2)
        np.testing.assert_allclose(r['sigma_cross'],np.sqrt((5*11+9)/100))
        with self.assertRaises(ValueError):
            signal_to_noise([1],[4],[9],[-3],[100],np.nan,0)

    def test_rebin(self):
        a = np.arange(8*6*4).reshape(8,6,4)
        b = rebin_mean(a,(4,3,2))
        self.assertEqual(b.shape,(4,3,2))
        np.testing.assert_allclose(a.mean(),b.mean())

    def test_invalid_geometry_and_empty_bins(self):
        a = np.ones((4,4,4))
        with self.assertRaises(ValueError):
            matched_power(a,a[:,:,:2],20,[0,1])
        p = matched_power(a,a,20,[0,.001,1])
        self.assertEqual(p['modes'][0],0)
        self.assertTrue(np.isnan(p['k'][0]))

if __name__ == '__main__':
    unittest.main()
