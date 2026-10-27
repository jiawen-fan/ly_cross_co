"""Periodic grid-aligned sparse skewer experiment, using exact pair compression."""
import argparse,json,os
from pathlib import Path
import numpy as np
import h5py
from lya_co.co.co_ly_sn import rebin_mean
from lya_co.estimator.compare import compare_estimators

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--co',required=True);p.add_argument('--lya',required=True)
p.add_argument('--nmesh',type=int,default=250)
p.add_argument('--sightlines',type=int,default=1000)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--box',type=float,default=250.)
p.add_argument('--kmax',type=float,default=1.)
p.add_argument('--rmax',type=float,default=80.)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
if not 1<=a.sightlines<=a.nmesh**2 or a.kmax>np.pi*a.nmesh/a.box/2 or a.kmax<=2*np.pi/a.box or a.rmax<=0:
    p.error('Invalid count/scales; kmax must be below half Nyquist and above fundamental')
a.output.mkdir(parents=True,exist_ok=False)
with np.load(a.co) as f: co=rebin_mean(f['map_cube'],(a.nmesh,)*3)
# Retain only randomly selected full z-columns. No reconstruction, no pixel noise.
rng=np.random.default_rng(a.seed)
ids=np.sort(rng.choice(a.nmesh**2,a.sightlines,replace=False))
x,y=np.unravel_index(ids,(a.nmesh,a.nmesh))
sparse=np.zeros(co.shape,dtype=float)
with h5py.File(a.lya,'r') as f:
    d=f['map']; native=np.asarray(d.shape)
    if native.shape!=(3,) or np.any(native%a.nmesh):
        raise ValueError('Native map dimensions must be integer multiples of nmesh')
    fx,fy,fz=native//a.nmesh
    # Each selected column is a finite-area block-averaged mock sightline.
    for i,j in zip(x,y):
        block=d[i*fx:(i+1)*fx,j*fy:(j+1)*fy,:]
        sparse[i,j]=block.reshape(fx,fy,a.nmesh,fz).mean(axis=(0,1,3),dtype=float)
mean=float(sparse[x,y].mean())
prob=a.sightlines/a.nmesh**2
sparse[x,y]=(sparse[x,y]-mean)/prob
edges=np.linspace(0,a.kmax,41)
r=compare_estimators(co,sparse,a.box,edges,r_max=a.rmax,window='cosine',run_pairs=False)
# radial_reference is EXACTLY the original weighted sparse pair monopole:
# integration volumes dV for CO, dV/prob for selected Ly-alpha pixels.
np.savez(a.output/'spectra.npz',**r,skewer_ix=x,skewer_iy=y,selection_probability=prob)
np.savetxt(a.output/'spectra.csv',np.column_stack([r['k'],r['radial_reference'],r['fft_windowed']]),delimiter=',',header='k,sparse_pair_monopole,windowed_sparse_fft',comments='')
config=vars(a).copy();config['output']=str(a.output)
config.update(sample_mean_delta_f=mean,selection_probability=prob,los_axis=2,
              backend='exact periodic grid pair compression',redshift='unverified: Ly-alpha filename says z2.5',
              workers=os.environ.get('OMP_NUM_THREADS','1'))
(a.output/'config.json').write_text(json.dumps(config,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
g=r['modes']>0
plt.plot(r['k'][g],r['radial_reference'][g],label='Sparse pair monopole')
plt.plot(r['k'][g],r['fft_windowed'][g],label='Windowed sparse FFT')
plt.xlabel('k [h/cMpc]');plt.ylabel('Cross power [microK (cMpc/h)^3]')
plt.axhline(0,color='gray');plt.legend();plt.tight_layout()
plt.savefig(a.output/'comparison.png',dpi=160)
print('Saved',a.output,flush=True)
