"""ASTRID spectra -> existing Dachshund -> three-method cross multipoles."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from lya_co.lya.astrid import load_maps, read_sightlines
from lya_co.reconstruction.dachshund import reconstruct
from lya_co.spectra.multipoles import fft_multipoles, power_1d, gaussian_truth_sn
from lya_co.estimator.sightline_pairs import sightline_pair_multipoles


def plot_comparison(k, spectra, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, (ax, ell) in enumerate(zip(axes, (0, 2, 4))):
        for name, power in spectra.items():
            ax.plot(k, power[i], label=name)
        ax.axhline(0, color='gray', lw=.5)
        ax.set(xlabel='k [h/cMpc]', ylabel=f'P_{ell} [microK (cMpc/h)^3]')
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output/'multipoles.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('co', 'lya-map', 'spectra', 'dachshund', 'output'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--mean-flux', type=float, required=True,
                   help='Use exactly the mean flux used to create the true map')
    p.add_argument('--nmesh', type=int, default=250)
    p.add_argument('--sightlines', type=int, default=1000)
    p.add_argument('--stride', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-seed', type=int, help='Omit for noiseless pixels; otherwise add sigma Gaussian noise')
    p.add_argument('--sigma', type=float, default=.02)
    p.add_argument('--tau-scale', type=float, default=1.)
    p.add_argument('--pixel-origin', type=float, choices=(0., .5), default=0.)
    p.add_argument('--kmax', type=float, default=1.)
    p.add_argument('--bins', type=int, default=40)
    p.add_argument('--rmax', type=float, default=80.)
    p.add_argument('--dr', type=float, default=.1)
    p.add_argument('--co-samples', type=int, default=20000)
    p.add_argument('--lperp', type=float, default=2.5)
    p.add_argument('--lpara', type=float, default=2.1)
    p.add_argument('--corr-variance', type=float, default=.05)
    p.add_argument('--pcg-tol', type=float, default=.01)
    p.add_argument('--pcg-max-iter', type=int, default=400)
    a = p.parse_args()
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    print('Reading ASTRID z sightlines', flush=True)
    sight = read_sightlines(a.spectra, a.sightlines, a.seed, a.mean_flux,
                           a.tau_scale, a.stride, a.sigma, a.noise_seed, a.pixel_origin)
    box = sight['box']
    if a.bins < 1 or not 2*np.pi/box < a.kmax <= np.pi*a.nmesh/box/2:
        p.error('Require fundamental < kmax <= half map Nyquist')
    if a.kmax > np.pi/sight['dz']:
        p.error('kmax exceeds the sampled sightline LOS Nyquist; reduce stride')
    np.savez(out/'sightlines.npz', **sight)
    np.savetxt(out/'sightline_ids.csv', sight['ids'], fmt='%d', header='fake_spectra_id')
    print('Running existing Dachshund executable', flush=True)
    rec = reconstruct(a.dachshund, sight['positions'], sight['delta'], a.sigma,
                      box, a.nmesh, out/'dachshund', a.lperp, a.lpara,
                      a.corr_variance, a.pcg_tol, a.pcg_max_iter)
    np.save(out/'reconstruction.npy', rec)
    print('Reading aligned full maps and calculating Fourier multipoles', flush=True)
    co, truth = load_maps(a.co, a.lya_map, a.nmesh)
    edges = np.linspace(0, a.kmax, a.bins+1)
    ground = fft_multipoles(co, truth, box, edges)
    reconstructed = fft_multipoles(co, rec, box, edges)
    print('Counting sparse sightline pairs', flush=True)
    pair = sightline_pair_multipoles(co, sight['positions'], sight['delta'], box,
                                    ground['k'], a.rmax, a.dr, a.co_samples, a.seed)
    powers = dict(truth=ground['power'], reconstruction=reconstructed['power'],
                  pairs=pair['power'])
    np.savez(out/'multipoles.npz', k=ground['k'], edges=edges, modes=ground['modes'],
             ells=ground['ells'], **powers)
    columns = [ground['k'], ground['modes']]
    headers = ['k', 'modes']
    for name, power in powers.items():
        for i, ell in enumerate((0, 2, 4)):
            columns.append(power[i]); headers.append(f'{name}_ell{ell}')
    np.savetxt(out/'multipoles.csv', np.column_stack(columns), delimiter=',',
               header=','.join(headers), comments='')
    # Auto multipoles and 1D spectra are separately labeled; noisy spectra retain noise.
    auto_truth = fft_multipoles(truth, truth, box, edges)
    auto_rec = fft_multipoles(rec, rec, box, edges)
    auto_co = fft_multipoles(co, co, box, edges)
    k1d, p1d = power_1d(sight['values'], box)
    np.savez(out/'auto_spectra.npz', k=ground['k'], co=auto_co['power'],
             truth=auto_truth['power'], reconstruction=auto_rec['power'],
             k1d=k1d, sightline_p1d=p1d)
    plot_comparison(ground['k'], powers, out)
    truth_sn = gaussian_truth_sn(co, truth, box, edges)
    np.savez(out/'truth_gaussian_sn.npz', k=ground['k'], **truth_sn)
    np.savetxt(out/'truth_gaussian_sn.csv', np.column_stack([ground['k'], truth_sn['sn'].T]),
               delimiter=',', header='k,sn_ell0,sn_ell2,sn_ell4', comments='')
    (out/'truth_gaussian_sn.json').write_text(json.dumps({
        'sn_total': truth_sn['sn_total'], 'type': 'full-map Gaussian sample variance only',
        'warning': 'not the covariance of reconstruction or sparse pairs'}, indent=2))
    config = vars(a).copy()
    config.update(box=box, pair_co_count=len(pair['co_ids']), pairs_used=pair['pairs_used'],
                  pair_window='cosine', pair_periodic=True,
                  reconstruction_boundary='nonperiodic upstream Dachshund',
                  redshift='Read input metadata; no redshift relabeling or conversion',
                  covariance='Use ensemble_sn.py with independent matching realizations',
                  executable_sha256=hashlib.sha256(Path(a.dachshund).expanduser().read_bytes()).hexdigest())
    rev = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
    config['git_revision'] = rev.stdout.strip() or 'unavailable'
    (out/'config.json').write_text(json.dumps(config, indent=2))
    np.savetxt(out/'pair_co_ids.csv', pair['co_ids'], fmt='%d', header='flat_co_cell_id')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for i, ax in enumerate(axes):
        ax.plot(ground['k'], truth_sn['sn'][i])
        ax.set(xlabel='k [h/cMpc]', ylabel=f'Ground truth S/N, ell={2*i}')
    fig.suptitle('Full-map Gaussian sample variance only')
    fig.tight_layout()
    fig.savefig(out/'truth_gaussian_sn.png', dpi=160)
    plt.close(fig)
    print('Saved', out, flush=True)


if __name__ == '__main__':
    main()
