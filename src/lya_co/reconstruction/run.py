from pathlib import Path
import subprocess
import numpy as np
from fake_spectra.plot_spectra import PlottingSpectra

# ---------- Settings ----------
spec_file = Path("/projects/pbreysse/lya_forest_CO/lya_storage/Lyman-alpha-ASTRID_z2.5_unzipped/Lyman-alpha-ASTRID_z2.5/spectra_ASTRID_noiseless_z2.5_1000_voxels.hdf5")
dachshund = Path("/users/jiawenfan/work/lya_CO/dachshund/dachshund.exe").resolve()

work = Path("astrid_reconstruction").resolve()
work.mkdir(exist_ok=True)

dperp = 2.5          # mean sightline spacing, cMpc/h
nmesh = 64         # reconstructed grid: 250^3
sigma_input = 0.02  # assumed delta_F uncertainty; no noise is added
tau_scale = 1.0     # use the SAME optical-depth scaling as your true map

# ---------- Read ASTRID spectra ----------
sp = PlottingSpectra(
    num=1,
    base="./",
    savedir=str(spec_file.parent),
    savefile=spec_file.name,
    res=None,
)

L = float(sp.box) / 1000.0  # ckpc/h -> cMpc/h

# Use only spectra along z (fake_spectra axis convention: 1, 2, 3).
available = np.flatnonzero(np.asarray(sp.axis) == 3)

n_skewers = 1000 

if n_skewers == 0:
    raise ValueError("No z-directed spectra found; check sp.axis.")

rng = np.random.default_rng(100)
ids = rng.choice(available, size=n_skewers, replace=False)

tau = sp.get_tau(elem="H", ion=1, line=1215)

# Mean flux from the full z-directed sample, not each individual skewer.
# If your true map uses a prescribed mean flux, use that same value here.
mean_flux = np.mean(np.exp(-tau_scale * tau[available]))
# delta_F = np.exp(-tau_scale * tau[ids]) / mean_flux - 1.0

positions = np.asarray(sp.cofm)[ids] / 1000.0
# nz = delta_F.shape[1]
# z = np.arange(nz) * L / nz  # matches Qezlou's pixel-position convention
#slided it more
delta_F = np.exp(-tau_scale * tau[ids]) / mean_flux - 1.0

# Original pixel coordinates
nz_native = delta_F.shape[1]
z = np.arange(nz_native) * L / nz_native

# Keep every fourth pixel and its original coordinate
stride = 5
delta_F = delta_F[:, ::stride]
z = z[::stride]
nz = len(z)
# ---------- Write Dachshund input ----------
# Binary float64 rows: x, y, z, sigma_deltaF, deltaF
pixel_file = work / "pixels.dat"

with open(pixel_file, "wb") as f:
    for i in range(n_skewers):
        pixels = np.column_stack([
            np.full(nz, positions[i, 0]),
            np.full(nz, positions[i, 1]),
            z,
            np.full(nz, sigma_input),
            delta_F[i],
        ])
        pixels.astype(np.float64).tofile(f)

# ---------- Write reconstruction configuration ----------
map_file = work / "map_reconstructed.dat"
config_file = work / "astrid.cfg"
#xhange perp to match the length
config_file.write_text(f"""\
lx = {L}
ly = {L}
lz = {L}
num_pixels = {n_skewers * nz}
map_nx = {nmesh}
map_ny = {nmesh}
map_nz = {nmesh}
corr_var_s = 0.05
corr_l_perp = 2.5 
corr_l_para = 2.1
pcg_tol = 1.0e-2 
pcg_max_iter = 400
pixel_data_path = {pixel_file}
map_path = {map_file}
""")

# ---------- Reconstruct ----------
subprocess.run(
    [str(dachshund), str(config_file)],
    check=True,
)

print("Reconstructed map:", map_file)