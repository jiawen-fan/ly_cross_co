"""Compare full-map FFT cross power with sparse grid-aligned sightline pairs."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import h5py
import numpy as np
from lya_co.co.co_ly_sn import rebin_mean
from lya_co.estimator.full_sparse import compare_full_sparse


def load_maps(co_path, lya_path, nmesh):
    shape = (nmesh,) * 3
    with np.load(co_path) as f:
        co = np.asarray(rebin_mean(f['map_cube'], shape))
    with h5py.File(lya_path, 'r') as f:
        # Materialize inside the file context even when no rebinning is needed.
        lya = np.asarray(rebin_mean(f['map'], shape))
    return co, lya


def plot_results(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = result['k']
    curves = [('full_fft', 'Full-map FFT'),
              ('sparse_pair_monopole', 'Sparse pair monopole'),
              ('full_pair_monopole', 'Full-map pair monopole'),
              ('full_windowed_fft', 'Windowed full-map FFT'),
              ('sparse_windowed_fft', 'Windowed sparse FFT')]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    finite = np.concatenate([np.abs(result[key][np.isfinite(result[key])]) for key, _ in curves])
    threshold = max(float(finite.max()) * 1e-3, 1e-12) if finite.size else 1.
    for ax in axes:
        for key, label in curves:
            ax.plot(k, result[key], label=label, lw=1.5)
        ax.axhline(0, color='gray', lw=.7)
        ax.set_xlabel('k [h/cMpc]')
        ax.set_ylabel('Cross power [microK (cMpc/h)^3]')
    axes[0].set_title('Linear scale')
    axes[1].set_yscale('symlog', linthresh=threshold)
    axes[1].set_title('Signed log scale')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / 'comparison.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, label in zip(axes, ['P_CO', 'P_F', 'r_full'],
                              ['CO auto [microK^2 (cMpc/h)^3]',
                               'Ly-alpha auto [(cMpc/h)^3]', 'Full-map correlation coefficient']):
        ax.plot(k, result[key])
        ax.set_xlabel('k [h/cMpc]')
        ax.set_ylabel(label)
        if key != 'r_full':
            ax.set_yscale('log')
    axes[2].set_ylim(-1.05, 1.05)
    fig.tight_layout()
    fig.savefig(output / 'full_map_diagnostics.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--co', required=True)
    p.add_argument('--lya', required=True)
    p.add_argument('--nmesh', type=int, default=250)
    p.add_argument('--sightlines', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--box', type=float, default=250.)
    p.add_argument('--kmax', type=float, default=1.)
    p.add_argument('--bins', type=int, default=40)
    p.add_argument('--rmax', type=float, default=80.)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if (not np.all(np.isfinite([a.box, a.kmax, a.rmax])) or a.box <= 0
            or a.nmesh < 2 or a.bins < 1 or a.rmax <= 0
            or not 1 <= a.sightlines <= a.nmesh**2
            or not 2*np.pi/a.box < a.kmax <= np.pi*a.nmesh/a.box/2):
        p.error('Invalid grid/count/scales: require fundamental < kmax <= half Nyquist')
    a.output.mkdir(parents=True, exist_ok=False)
    print('Loading and block-averaging full maps', flush=True)
    co, lya = load_maps(a.co, a.lya, a.nmesh)
    print('Comparing full-map FFT and sparse pairs', flush=True)
    result = compare_full_sparse(co, lya, a.box, np.linspace(0, a.kmax, a.bins+1),
                                 a.sightlines, a.seed, a.rmax)
    np.savez(a.output / 'spectra.npz', **result)
    columns = ['k', 'modes', 'full_fft', 'sparse_pair_monopole',
               'full_pair_monopole', 'full_windowed_fft', 'sparse_windowed_fft',
               'P_CO', 'P_F', 'r_full']
    np.savetxt(a.output / 'spectra.csv', np.column_stack([result[c] for c in columns]),
               delimiter=',', header=','.join(columns), comments='')
    np.savetxt(a.output / 'sightlines.csv',
               np.column_stack([result['skewer_ix'], result['skewer_iy']]),
               fmt='%d', delimiter=',', header='ix,iy', comments='')
    config = vars(a).copy()
    config['output'] = str(a.output)
    for key in ['sample_mean_delta_f', 'full_mean_delta_f', 'selection_probability']:
        config[key] = result[key]
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    config.update(git_revision=revision.stdout.strip() or 'unavailable',
                  los_axis=2, window='cosine', backend='exact periodic grid pair compression',
                  redshift='unverified; supplied Ly-alpha filename says z2.5',
                  workers=os.environ.get('OMP_NUM_THREADS', '1'),
                  units={'k': 'h/cMpc', 'box': 'cMpc/h', 'cross': 'microK (cMpc/h)^3'},
                  lya_input='delta_F; no additional flux normalization',
                  covariance='not estimated for sparse pairs')
    (a.output / 'config.json').write_text(json.dumps(config, indent=2))
    plot_results(result, a.output)
    print('Saved', a.output, flush=True)


if __name__ == '__main__':
    main()
