# Ly-alpha and CO Cross-Correlation

Research pipeline for cross-correlating Ly-alpha forest
observations and CO line-intensity maps using simulations.

## Main Components
- Ly-alpha mock spectra
- CO intensity mapping
- Ly-alpha tomographic reconstruction
- HIgh-k Power SpecTrum and bispectrum EstimatoR
- Auto- and cross-power spectra
- Numerical verification and testing

## Computation
Scientific computations run on SMU M3 HPC.
Large datasets are stored outside this repository.

## Development
Source code is version-controlled with GitHub.
AI-assisted development takes place outside HPC.

##Code structure:
Pipeline structure



CO Map making: 
CO mapmaking, CO power spectrum generating
Functions:
1. Read in simulation
 	Load in halo mass, positions, velocities information.
2. Assign sfr to each halo: 
using behroozi table, assign sft based on log(halo mass).
3. Sfr to luminosity:
	SFR -> IR -> CO(1-0) luminosity in Li 2016, consider adding Chung 2019
4. Real space map making:
	Map making, painting luminosity and converting to CO temperature.CIC(cloud in cell) optional.
5. Map saving:
	Save map in ‘npz’ or ‘.hdf5’ files.
6. Calculating auto power spectrum:
	Spherical auto/cross P(k); k in h/cMpc, P in (cMpc/h)^3.

Lyman Alpha map reconstruction:
In construction
Noise input for realistic Ly alpha (if needed later)
Reading astrid.
Running dachshund input 
Reconstruction out put, rebinning the map for cross correlation
Out the map


Lyman Alpha map post processing:
In construction
1. Converting from delt F to Flux, and vise versa
2. ? Masking HCD.
3. Calculate 1d power spectrum 
4. Calculated spherical average power spectrum
5. Calculated power spectrum with multipole moments.


HIgh-k Power SpecTrum and bispectrum EstimatoR on CO and Lyman Alpha:
1. use the map generated.
2. follow Hipster code, instead of auto cross, make cross between co and lyman alpha with sparse sightlines
3. Calculating auto power spectrum:
	Spherical auto/cross P(k); k in h/cMpc, P in (cMpc/h)^3.


Cross power spectrum and S/N
Calculating signal to noise ratio for CO and lyman alpha
Calculating the cross power spectrum with different input maps
Comparing results
