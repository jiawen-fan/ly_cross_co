#!/usr/bin/env python3
"""Small-grid HPC comparison; use --help. No network calls or AI clients."""
import argparse
import json
from pathlib import Path
import time
import subprocess
import numpy as np
from lya_co.co.co_ly_sn import rebin_mean
from lya_co.estimator.compare import compare_estimators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--co', type=Path, help='CO npz containing map_cube, microK')
    parser.add_argument('--lya', type=Path, help='HDF5 delta_F map, already normalized')
    parser.add_argument('--lya-key', default='map')
    parser.add_argument('--redshift', type=float, default=2.6,
                        help='shared input snapshot redshift; records provenance, does not transform maps')
    parser.add_argument('--boxsize', type=float, nargs=3, default=[250.,250.,250.])
    parser.add_argument('--nmesh', type=int, default=8, help='whole-box block average, NOT a cutout')
    parser.add_argument('--bins', type=int, default=8)
    parser.add_argument('--kmax', type=float, help='default: half the smallest grid Nyquist')
    parser.add_argument('--r-max', type=float, help='cMpc/h; default: no cutoff')
    parser.add_argument('--window', choices=['tophat','cosine'], default='tophat')
    parser.add_argument('--skip-pairs', action='store_true', help='use fast periodic references only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, required=True, help='new output directory')
    args = parser.parse_args()
    if not np.isfinite(args.redshift) or args.redshift < 0:
        parser.error('redshift must be finite and nonnegative')
    if bool(args.co) != bool(args.lya):
        parser.error('Supply both --co and --lya, or neither for synthetic data')
    if args.nmesh < 4 or args.bins < 1 or min(args.boxsize) <= 0:
        parser.error('nmesh >= 4, bins >= 1 and positive box lengths required')
    if not args.skip_pairs and args.nmesh**6 > 2_000_000:
        parser.error('Old estimator is O(nmesh^6): use nmesh <= 11 or --skip-pairs')
    if args.window == 'cosine' and args.r_max is None:
        parser.error('--window cosine requires --r-max')
    limit = .5 * min(np.pi*args.nmesh/np.asarray(args.boxsize))
    kmax = limit if args.kmax is None else args.kmax
    if not np.isfinite(kmax) or not 0 < kmax <= limit:
        parser.error('kmax must be positive and <= half the smallest grid Nyquist')
    if kmax <= 2*np.pi/max(args.boxsize):
        parser.error('kmax must exceed the smallest nonzero Fourier mode')
    args.output.mkdir(parents=True, exist_ok=False)
    shape = (args.nmesh,)*3
    if args.co:
        with np.load(args.co, allow_pickle=False) as data:
            co = rebin_mean(data['map_cube'], shape)
        import h5py
        with h5py.File(args.lya, 'r') as data:
            delta_f = rebin_mean(data[args.lya_key], shape)
    else:
        rng = np.random.default_rng(args.seed)
        co = rng.normal(size=shape)
        delta_f = -.7*co + .3*rng.normal(size=shape)
    edges = np.linspace(0, kmax, args.bins+1)
    start = time.perf_counter()
    result = compare_estimators(co, delta_f, args.boxsize, edges,
        r_max=args.r_max, window=args.window, run_pairs=not args.skip_pairs)
    elapsed = time.perf_counter()-start
    good = result['modes'] > 0
    checks = {}
    if not args.skip_pairs:
        # Scale-aware absolute tolerance avoids unstable ratios near zero crossings.
        scale = max(np.max(abs(result['radial_reference'][good])), 1e-30)
        error = np.max(abs(result['pair_cross'][good]-result['radial_reference'][good]))/scale
        checks['pair_vs_radial_max_scaled_error'] = float(error)
        checks['pair_vs_radial_pass'] = bool(error < 1e-10)
    if args.r_max is None:
        scale = max(np.max(abs(result['fft_cross'][good])), 1e-30)
        error = np.max(abs(result['fft_windowed'][good]-result['fft_cross'][good]))/scale
        checks['untapered_fft_max_scaled_error'] = float(error)
        checks['untapered_fft_pass'] = bool(error < 1e-10)
    np.savez(args.output/'spectra.npz', **result)
    names = ['k','modes','fft_cross','fft_windowed','radial_reference','pair_cross']
    np.savetxt(args.output/'spectra.csv', np.column_stack([result[n] for n in names]),
               delimiter=',', header=','.join(names), comments='')
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[1], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = 'unavailable'
    config.update(git_commit=revision, seconds=elapsed, checks=checks,
                  input_quantity_lya='delta_F', input_units_co='microK',
                  input_geometry='aligned complete periodic xyz grids',
                  numpy_version=np.__version__)
    (args.output/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,5))
    for key, label, style in [('fft_cross','Discrete FFT shells','-o'),
                              ('fft_windowed','FFT shells with separation window','--'),
                              ('radial_reference','Continuous-angle radial reference','-'),
                              ('pair_cross','Old pair estimator','x')]:
        ax.plot(result['k'][good],result[key][good],style,label=label)
    ax.axhline(0,color='gray',lw=.7)
    ax.set(xlabel='k [h/cMpc]',ylabel='Cross power [microK (cMpc/h)^3]')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output/'comparison.png',dpi=160)
    plt.close(fig)
    print(json.dumps(checks,indent=2))
    print(f'Saved {args.output}; estimator time {elapsed:.2f} seconds')
    if any(value is False for key,value in checks.items() if key.endswith('_pass')):
        raise SystemExit('Comparison identity failed; inspect outputs')


if __name__ == '__main__':
    main()
