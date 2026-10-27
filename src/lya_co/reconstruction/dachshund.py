"""File adapter for the existing caseywstark Dachshund executable, not a solver."""
from pathlib import Path
import subprocess
import numpy as np


def reconstruct(executable, positions, delta, sigma, box, nmesh, directory,
                lperp=2.5, lpara=2.1, variance=.05, tolerance=.01, max_iter=400):
    work = Path(directory).resolve()
    work.mkdir(parents=True, exist_ok=False)
    executable = Path(executable).expanduser().resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    positions, delta = np.asarray(positions), np.asarray(delta)
    if positions.shape != (delta.size, 3) or not np.all(np.isfinite(positions)) or not np.all(np.isfinite(delta)):
        raise ValueError('Expected finite (N,3) positions and (N,) delta')
    if min(sigma, box, nmesh, lperp, lpara, variance, tolerance, max_iter) <= 0:
        raise ValueError('Dachshund scales and sigma must be positive')
    # Short relative names also avoid the upstream config parser's small buffers.
    pixels = np.column_stack([positions, np.full(delta.size, sigma), delta])
    pixels.astype('=f8').tofile(work / 'pixels.bin')
    config = dict(lx=box, ly=box, lz=box, num_pixels=delta.size,
                  map_nx=nmesh, map_ny=nmesh, map_nz=nmesh,
                  corr_var_s=variance, corr_l_perp=lperp, corr_l_para=lpara,
                  pcg_tol=tolerance, pcg_max_iter=max_iter,
                  pixel_data_path='pixels.bin', map_path='map.bin')
    (work/'input.cfg').write_text(''.join(f'{k} = {v}\n' for k, v in config.items()))
    with (work/'dachshund.log').open('w') as log:
        subprocess.run([str(executable), 'input.cfg'], cwd=work, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    log_text = (work/'dachshund.log').read_text()
    if 'PCG reached max iteration before stop condition' in log_text:
        raise RuntimeError('Dachshund PCG did not converge; inspect dachshund.log')
    raw = np.fromfile(work/'map.bin', dtype='=f8')
    if raw.size != nmesh**3 or not np.all(np.isfinite(raw)):
        raise ValueError('Dachshund output is missing, nonfinite or has the wrong grid size')
    # Upstream util.cc: cell centers, x outermost, z fastest.
    return raw.reshape((nmesh,)*3)
