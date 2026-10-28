"""Calculate reconstruction/pair S/N from matching independent realizations."""
import argparse
import json
from pathlib import Path
import numpy as np
from lya_co.spectra.covariance import ensemble_sn


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--realizations', type=Path, nargs='+', required=True)
    p.add_argument('--kind', choices=['noise-only', 'independent-mocks'], required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    paths = [path.resolve() for path in a.realizations]
    if len(set(paths)) != len(paths) or a.reference.resolve() in paths:
        p.error('Realizations must be distinct and exclude the reference')
    a.output.mkdir(parents=True, exist_ok=False)
    ref = np.load(a.reference/'multipoles.npz')
    configs = [json.loads((path/'config.json').read_text()) for path in paths]
    ref_config = json.loads((a.reference/'config.json').read_text())
    keys = ['lya_map_field', 'nmesh', 'sightlines', 'stride', 'seed', 'mean_flux', 'sigma', 'tau_scale',
            'pixel_origin', 'kmax', 'bins', 'rmax', 'dr', 'co_samples', 'lperp',
            'lpara', 'corr_variance', 'pcg_tol', 'pcg_max_iter', 'box', 'executable_sha256']
    for config in configs:
        if any(config[key] != ref_config[key] for key in keys):
            p.error('Ensemble settings do not match the reference')
    if a.kind == 'noise-only':
        seeds = [c['noise_seed'] for c in configs]
        if None in seeds or len(set(seeds)) != len(seeds):
            p.error('Noise realizations need distinct non-null noise seeds')
        if any(any(c[key] != ref_config[key] for key in ['co', 'lya_map', 'spectra']) for c in configs):
            p.error('Noise-only ensemble must use the same input fields')
    if a.kind == 'independent-mocks':
        inputs = [tuple(c[key] for key in ['co', 'lya_map', 'spectra']) for c in configs]
        if len(set(inputs)) != len(inputs):
            p.error('Independent-mocks requires distinct input field sets, not repeated noise draws')
    methods = ['reconstruction', 'pairs'] if a.kind == 'noise-only' else ['truth', 'reconstruction', 'pairs']
    samples = {name: [] for name in methods}
    for path in paths:
        with np.load(path/'multipoles.npz') as data:
            for key in ['k', 'edges', 'ells']:
                np.testing.assert_allclose(data[key], ref[key], equal_nan=True)
            for name in methods:
                samples[name].append(data[name])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    report = dict(kind=a.kind, realizations=[str(p) for p in paths],
                  interpretation=('Conditional pixel-noise S/N at fixed ASTRID fields and sightlines; excludes cosmic variance and sampling variance'
                                  if a.kind == 'noise-only' else 'Empirical S/N; independent mock fields required'))
    for name in methods:
        result = ensemble_sn(ref[name], np.asarray(samples[name]))
        np.savez(a.output/f'{name}_sn.npz', k=ref['k'], **result)
        report[name] = {'sn_total': result['sn_total'], 'nmock': result['nmock'], 'hartlap': result['hartlap']}
        np.savetxt(a.output/f'{name}_sn.csv', np.column_stack([ref['k'], result['sn'].T]),
                   delimiter=',', header='k,sn_ell0,sn_ell2,sn_ell4', comments='')
        for i, ax in enumerate(axes):
            ax.plot(ref['k'], result['sn'][i], label=name)
            ax.set(xlabel='k [h/cMpc]', ylabel=f'S/N, ell={2*i}')
    axes[0].legend()
    fig.suptitle(a.kind+' covariance')
    fig.tight_layout()
    fig.savefig(a.output/'sn_multipoles.png', dpi=160)
    (a.output/'sn_summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
