import numpy as np
from astropy.cosmology import Planck15
from lim_mocker_cartesian import (
    load_astrid, behroozi_sfr, li_luminosity, make_map, power_spectrum, save_map,
)

# Configuration: use the same snapshot and grid as your Ly-alpha map.
CATALOG = "/projects/pbreysse/lya_forest_CO/lya_storage/Subhalo"
OUTPUT = "co_li_cartesian.npz"
POWER_OUTPUT = "co_li_power.npz"
Z = 2.6
HUBBLE_H = Planck15.h
H_Z = Planck15.H(Z).value             # km/s/Mpc
BOXSIZE = 250.                        # cMpc/h
NMESH = (1000, 1000, 1000)
MIN_MASS = 5e10                       # physical Msun; choose your completeness cut
VELOCITY_FACTOR = 1.                  # raw velocity -> physical peculiar km/s
USE_RSD = True
LOS = 2                              # x=0, y=1, z=2
ORIGIN = (0., 0., 0.)
GRID_OFFSET = .5                      # .5: cell centers; 0: grid nodes
SEED = 42
SIGMA_SFR = .3
SIGMA_CO = .3


def main():
    # 1. Read the catalog and convert its units.
    halos = load_astrid(CATALOG, HUBBLE_H, MIN_MASS, velocity_factor=VELOCITY_FACTOR)

    # 2. Li model: halo mass -> mean SFR -> IR -> CO luminosity.
    sfr = behroozi_sfr(halos["mass"], Z)
    luminosity = li_luminosity(sfr, sigma_sfr=SIGMA_SFR, sigma_co=SIGMA_CO, seed=SEED)

    # 3. Shift along the LOS if requested, then paint the full periodic box.
    temperature, metadata = make_map(
        halos["positions"], luminosity, BOXSIZE, NMESH, Z, HUBBLE_H, H_Z,
        velocities=halos["velocities"] if USE_RSD else None,
        los=LOS, origin=ORIGIN, grid_offset=GRID_OFFSET,
    )


    metadata.update(catalog=CATALOG, min_mass_msun=MIN_MASS,
                    sfr_source="Behroozi2013", seed=SEED,
                    alpha=1.37, beta=-1.74, delta_mf=1.,
                    sigma_sfr_dex=SIGMA_SFR, sigma_co_dex=SIGMA_CO)
    save_map(OUTPUT, temperature, metadata)

    # 4. Measure the CO auto spectrum. Keep CIC smoothing and shot noise.
    pk = power_spectrum(temperature, BOXSIZE)
    with open(POWER_OUTPUT, "xb") as f:
        np.savez(f, **pk)
    print("Saved", OUTPUT, "and", POWER_OUTPUT)
    print("Mean CO temperature [microK]:", temperature.mean())
    return temperature, metadata, pk


if __name__ == "__main__":
    main()
