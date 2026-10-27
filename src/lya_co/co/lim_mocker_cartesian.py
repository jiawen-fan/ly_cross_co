"""Li-model CO(1-0) mocks in a periodic Cartesian box.

Units: positions/box cMpc/h, mass Msun, SFR Msun/yr, velocities km/s,
luminosity Lsun, temperature microK. Array axes are always (x, y, z).
Start with example_cartesian.ipynb. There are no model classes or CLI.
"""
from itertools import product
from pathlib import Path
import json
import numpy as np
from astropy import constants as const, units as u
from scipy.fft import rfftn, fftfreq, rfftfreq
from scipy.interpolate import RegularGridInterpolator

from astropy import units as u
from astropy.cosmology import Planck18, z_at_value

__version__ = "3.0.0"


def load_peak_patch(
    path,
    h,
    min_mass=5e10,
    velocity_factor=1.0,
    center_redshift=2.6,
    boxsize_cmpch=None,
    los_axis=2,
    cosmo=Planck18,
):
    """Load halos and cut a box centered at center_redshift.

    Assumes catalog x, y, z are absolute comoving positions in cMpc.
    Returned positions and box lengths are in cMpc/h. The box depth along
    los_axis is half the corresponding boxsize_cmpch length.
    """
    with np.load(path) as halo_info:
        mass = halo_info["M"].reshape(-1)
        z_halo = halo_info["zhalo"].reshape(-1)

        positions = h * np.column_stack(
            (halo_info["x"], halo_info["y"], halo_info["z"])
        )
        velocities = velocity_factor * np.column_stack(
            (halo_info["vx"], halo_info["vy"], halo_info["vz"])
        )

        keep = mass > min_mass

        if boxsize_cmpch is not None:
            box = np.broadcast_to(
                np.asarray(boxsize_cmpch, dtype=float), (3,)
            ).copy()

            # Make the total line-of-sight depth half the supplied box length.
            box[los_axis] /= 2

            # # Center x and y on the catalog footprint; center the LOS at z=2.6.
            # box_center = 0.5 * (
            #     positions.min(axis=0) + positions.max(axis=0)
            # )
            chi_center = (
                cosmo.comoving_distance(center_redshift).to_value(u.Mpc) * h
            )
            # print(chi_center)
            # halo_z_cut = halo_info["z"] > (chi_center - boxsize/2/h) & halo_info["z"] < (chi_center + oxsize/2/h)
            # halo_x_cut =  halo_info["x"] > (-boxsize/2/h) & halo_info["z"] < (chi_center + 250/h)
            # box_center[los_axis] = chi_center
            # box_origin = box_center - box / 2
            # print(box_origin)
            # inside_box = np.all(
            #     (positions >= box_origin)
            #     & (positions < box_origin + box),
            #     axis=1,
            # )
            
            # keep &= inside_box

            # Apply the corresponding redshift interval around z=2.6.
            z_min = z_at_value(
                cosmo.comoving_distance,
                (chi_center - boxsize_cmpch / 2) / h * u.Mpc,
            ).value
            z_max = z_at_value(
                cosmo.comoving_distance,
                (chi_center + boxsize_cmpch / 2) /h * u.Mpc,
            ).value
            keep &= (z_halo >= z_min) & (z_halo <= z_max)
            keep &= (positions[:, 0] >= -boxsize_cmpch/2) &(positions[:, 0] <= boxsize_cmpch/2) 
            keep &= (positions[:, 1] >= -boxsize_cmpch/2) &(positions[:, 1] <= boxsize_cmpch/2) 
        z = positions[keep, 2]
        value = (z.max() + z.min()) / 2
        positions[keep, 2] -= value
        keep &= (positions[:, 2] >= -boxsize_cmpch/2) &(positions[:, 2] <= boxsize_cmpch/2) 
        print(min((positions[keep])[:,2]))
        print(max((positions[keep])[:,2]))
        print(min((positions[keep])[:,0]))
        print(max((positions[keep])[:,0]))
        print(min((positions[keep])[:,1]))
        print(max((positions[keep])[:,1]))
        positions[keep, 0] += 125
        positions[keep, 1] += 125
        positions[keep, 2] += 125

        # print(positions.min(axis=1),positions.max(axis=1))
        # print(positions.min(axis=2),positions.max(axis=2))
        print(len(mass[keep]))

        return {
            "mass": mass[keep],
            "positions": positions[keep],
            "velocities": velocities[keep],
            "redshift": z_halo[keep],
            "zformation": halo_info["zform"].reshape(-1)[keep],
        }
    
    
def load_astrid(path, h, min_mass=5e10, read_sfr=False, velocity_factor=1.):
    """Read an ASTRID Subhalo BigFile into memory and apply M > min_mass.

    Assumes SubhaloPos in ckpc/h and SubhaloMass in 1e10 Msun/h.
    velocity_factor converts SubhaloVel to physical peculiar km/s.
    Optional SubhaloSFR must be in Msun/yr. Edit keys here for another export.
    """
    from bigfile import BigFile
    with BigFile(str(path)) as catalog:
        mass = catalog["SubhaloMass"][:].reshape(-1) * 1e10 / h
        keep = mass > min_mass
        data = {
            "mass": mass[keep],
            "positions": catalog["SubhaloPos"][:][keep] / 1000.,
            "velocities": catalog["SubhaloVel"][:][keep] * velocity_factor,
            
        }
        print(len(mass[keep]))

        print(max(catalog["SubhaloPos"][:][keep][:,0] ))
        if read_sfr:
            data["sfr"] = catalog["SubhaloSFR"][:].reshape(-1)[keep]
    return data


def behroozi_sfr(mass_msun, z, table_path=None):
    """Mean SFR(M,z), interpolating the bundled Behroozi table as in LIMLAM.

    Interpolate linear SFR on log10(M/Msun), log10(1+z) coordinates.
    SciPy raises for masses/redshifts outside the table; no extrapolation.
    """
    path = table_path or Path(__file__).parent / "tables/sfr_behroozi_release.dat"
    table = np.loadtxt(path)
    logm, logz = np.unique(table[:, 1]), np.unique(np.log10(table[:, 0]))
    sfr_grid = np.empty((len(logm), len(logz)))
    im = np.searchsorted(logm, table[:, 1])
    iz = np.searchsorted(logz, np.log10(table[:, 0]))
    sfr_grid[im, iz] = 10.**table[:, 2]
    interpolate = RegularGridInterpolator((logm, logz), sfr_grid)
    mass_msun = np.atleast_1d(mass_msun)
    points = np.column_stack((np.log10(mass_msun), np.full(mass_msun.size, np.log10(1 + z))))
    return interpolate(points)


def li_luminosity(sfr, sigma_sfr=.3, sigma_co=.3, seed=42,
                  alpha=1.37, beta=-1.74, delta_mf=1.):
    """SFR -> IR -> CO(1-0) luminosity in Lsun (Li 2016 / Chung 2019).

    For Behroozi mean SFR, use the default two independent 0.3-dex scatters.
    For measured simulation SFR, set sigma_sfr=0 to avoid adding SFR scatter.
    Each stage preserves its input's linear mean. Seed fixes this array's
    realization; reordering/subsetting it changes the assignment of draws.
    """
    rng = np.random.default_rng(seed)

    def scatter(values, dex):
        sigma = dex * np.log(10.)
        return values * np.exp(rng.normal(0., sigma, values.shape) - sigma**2 / 2.)

    sfr = scatter(np.asarray(sfr, dtype=float), sigma_sfr)
    lir = sfr * 1e10 / delta_mf                              # Lsun
    lprime = np.zeros_like(lir)
    positive = lir > 0
    lprime[positive] = 10.**((np.log10(lir[positive]) - beta) / alpha)  # K km/s pc^2
    return scatter(4.9e-5 * lprime, sigma_co)                # Lsun


def make_map(positions, luminosity, boxsize, nmesh, z, h, H_z,
             velocities=None, los=2, origin=(0., 0., 0.), grid_offset=.5):
    """Paint luminosity with periodic CIC, then convert to CO temperature.

    H_z is H(z) in km/s/Mpc. Pass physical peculiar velocities for RSD;
    omit them for real space. Positions must initially be in real space.
    grid_offset=0.5 gives cell centers; 0 gives grid nodes. LOS: x=0,y=1,z=2.
    Returns (temperature, metadata). No beam, noise, or CIC deconvolution.
    """
    box = np.broadcast_to(np.asarray(boxsize, dtype=float), (3,))
    shape = np.broadcast_to(np.asarray(nmesh, dtype=int), (3,))
    dx = box / shape
    print(dx)
    luminosity = np.asarray(luminosity, dtype=float)
    grid = np.zeros(tuple(shape), dtype=float)

    # CIC: share each source between its eight neighbouring grid points.
    # Small batches keep the temporary coordinate/weight arrays manageable.
    # for start in range(0, len(luminosity), 250_000):
    #     sl = slice(start, start + 250_000)
    #     pos = np.array(positions[sl], dtype=float) - np.asarray(origin)
    #     if velocities is not None:
    #         pos[:, los] += h * (1 + z) / H_z * np.asarray(velocities[sl])[:, los]
    #     coordinate = (pos % box) / dx - grid_offset
    #     lower = np.floor(coordinate).astype(int)
    #     fraction = coordinate - lower
    #     for corner in product((0, 1), repeat=3):
    #         index = (lower + corner) % shape
    #         weight = np.prod(np.where(corner, fraction, 1 - fraction), axis=1)
    #         np.add.at(grid, tuple(index.T), luminosity[sl] * weight)


    #this is like the oringla Limmocker, which assigns to 8 neigboring cells
    # Assign each halo entirely to its nearest grid cell.
    for start in range(0, len(luminosity), 250_000):
        sl = slice(start, start + 250_000)
        pos = np.array(positions[sl], dtype=float) - np.asarray(origin)
    
        if velocities is not None:
            pos[:, los] += (
                h * (1 + z) / H_z * np.asarray(velocities[sl])[:, los]
            )
    
        coordinate = (pos % box) / dx - grid_offset
        index = np.floor(coordinate + 0.5).astype(int) % shape
    
        np.add.at(grid, tuple(index.T), luminosity[sl])
    # Tb = c^3 (1+z)^2 rho_L / [8 pi k_B nu_rest^3 H(z)].
    # Convert Lsun/(cMpc/h)^3 to SI; apply h^3 exactly once, then K -> microK.
    mpc = u.Mpc.to(u.m)
    H_si = H_z * 1000. / mpc
    conversion = (const.c.value**3 * (1 + z)**2 * const.L_sun.value * h**3 * 1e6
                  / (8 * np.pi * const.k_B.value * (115.27e9)**3 * H_si * mpc**3))
    temperature = grid * conversion / np.prod(dx)
    metadata = {
        "model": "Li2016", "code_version": __version__,
        "z": float(z), "h": float(h), "H_z_km_s_Mpc": float(H_z),
        "boxsize_cmpch": box.tolist(), "nmesh": shape.tolist(),
        "origin_cmpch": list(origin), "grid_offset": grid_offset, "axis_order": "xyz",
        "space": "redshift" if velocities is not None else "real", "los": los,
        "assignment": "ngp", "temperature_unit": "microK", "nu_rest_GHz": 115.27,
        "nsource": len(luminosity), "mean_microK": float(temperature.mean()),
        "point_source_shot_microK2_cmpch3": float(conversion**2 * np.sum(luminosity**2) / np.prod(box)),
    }
    return temperature, metadata


def power_spectrum(field_a, boxsize, field_b=None, nbins=30, k_edges=None):
    """Spherical auto/cross P(k); k in h/cMpc, P in [a][b](cMpc/h)^3.

    Subtract means, retain signed cross power, and omit the zero mode.
    Inputs must share the same grid. A delta_F input is used directly.
    Counts include both +/- modes. No shot subtraction or window correction.
    """
    a = np.asarray(field_a)
    b = a if field_b is None else np.asarray(field_b)
    if a.shape != b.shape:
        raise ValueError("Cross-power fields must have the same shape")
    box = np.broadcast_to(np.asarray(boxsize, dtype=float), (3,))
    dx = box / np.asarray(a.shape)
    fa = rfftn(a)
    fb = fa if field_b is None else rfftn(b)
    power = (fa * fb.conj()).real * np.prod(box) / a.size**2

    kx = 2 * np.pi * fftfreq(a.shape[0], dx[0])
    ky = 2 * np.pi * fftfreq(a.shape[1], dx[1])
    kz = 2 * np.pi * rfftfreq(a.shape[2], dx[2])
    # k = np.sqrt(kx**2 + ky**2 + kz**2).ravel()

    # rfftn stores only kz >= 0. Restore the missing negative-kz mode counts.
    # weight_z = np.full(fa.shape[2], 2.)
    # weight_z[0] = 1.
    # if a.shape[2] % 2 == 0:
    #     weight_z[-1] = 1.
    # weight = np.broadcast_to(weight_z, fa.shape).ravel()
    # keep = k > 0
    # if k_edges is None:
    #     k_edges = np.linspace(0., np.min(np.pi / dx), nbins + 1)
    # counts = np.histogram(k[keep], k_edges, weights=weight[keep])[0]
    # sum_k = np.histogram(k[keep], k_edges, weights=(k * weight)[keep])[0]
    # sum_p = np.histogram(k[keep], k_edges, weights=(power.ravel() * weight)[keep])[0]
    # mean_k = np.divide(sum_k, counts, out=np.full_like(sum_k, np.nan), where=counts > 0)
    # mean_p = np.divide(sum_p, counts, out=np.full_like(sum_p, np.nan), where=counts > 0)

    kgrid     = np.sqrt(sum(ki**2 for ki in np.meshgrid(kx,ky,kz,indexing='ij')))
    dk        = max(np.diff(kx)[0],np.diff(ky)[0],np.diff(kz)[0])

    kmax_dk   = int(np.ceil(max(np.amax(kx),np.amax(ky),np.amax(kz))/dk))
    kbins     = np.linspace(0,kmax_dk*dk,kmax_dk+1)

    Pk_nmodes = np.histogram(kgrid[kgrid>0],bins=kbins,weights=power[kgrid>0])[0]
    nmodes    = np.histogram(kgrid[kgrid>0],bins=kbins)[0]

    Pk = Pk_nmodes/nmodes
    k  = (kbins[1:]+kbins[:-1])/2

    return {"k": k, "Pk": Pk, "nmodes": nmodes.astype(int)}


def save_map(filename, temperature, metadata, overwrite=False):
    """Save the map, coordinates and JSON metadata as NPZ or HDF5."""
    filename = Path(filename)
    coordinates = {
        f"map_{axis}_cmpch": metadata["origin_cmpch"][i]
        + (np.arange(metadata["nmesh"][i]) + metadata["grid_offset"])
        * metadata["boxsize_cmpch"][i] / metadata["nmesh"][i]
        for i, axis in enumerate("xyz")
    }
    if filename.suffix == ".npz":
        with open(filename, "wb" if overwrite else "xb") as f:
            np.savez(f, map_cube=temperature, metadata_json=json.dumps(metadata), **coordinates)
    elif filename.suffix in (".h5", ".hdf5"):
        import h5py
        with h5py.File(filename, "w" if overwrite else "x") as f:
            f.create_dataset("map_cube", data=temperature)
            f.attrs["metadata_json"] = json.dumps(metadata)
            for key, value in coordinates.items():
                f.create_dataset(key, data=value)
    else:
        raise ValueError("Use .npz, .h5 or .hdf5")
