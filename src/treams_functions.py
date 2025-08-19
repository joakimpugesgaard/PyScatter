import treams
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
from src.focusedBeam import *
from src.domain import *
from src.interaction import *

def focusedBeam_treams(focused_beam, spheres, positions=None, radii=None):
    """
    Interface function between focusedBeam object and treams spheres.
    
    Parameters:
    -----------
    focused_beam : focusedBeam
        The focused beam object containing beam coefficients
    spheres : list or treams.TMatrix
        List of treams sphere TMatrix objects or single sphere TMatrix
    positions : array_like, optional
        Positions of spheres. (N, 3)dimensional. If None, assumes single sphere at origin
    radii : array_like, optional
        Radii of spheres
        
    Returns:
    --------
    tm_filtered : treams.TMatrix
        Filtered T-matrix for the specific m value
    coeffs : treams.core.PhysicsArray
        Incident field coefficients
    valid : np.ndarray or None
        Valid points mask if grid and radii are provided, else None
    """
    #parity basis everywhere
    treams.config.POLTYPE = "parity"
    #regular modes
    treams.config.MODETYPE = "regular"
        
    
    wl = focused_beam.wl * 1000 # convert to nm
    k = 2 * np.pi / wl  # wavenumber in 1/nm
    mz = focused_beam.mz 
    
    # Set default values
    if positions is None:
        positions = [[0, 0, 0]]  # Single sphere at origin
    
    # Handle both single sphere and cluster cases
    if isinstance(spheres, list):
        n_positions = len(positions)
        if n_positions != len(spheres):
            raise ValueError("Number of positions does not match number of spheres.")
        else:
            tm = treams.TMatrix.cluster(spheres, positions).interaction.solve()
    else:
        n_positions = 1
        tm = spheres
    
    
    # Define the spherical wave basis
    maxJ = focused_beam.maxJ
    basis = treams.SphericalWaveBasis.default(lmax=maxJ, nmax=n_positions, positions=np.zeros_like(positions))
    # Filter the spherical wave basis based on the m value of the focused beam
    basis = basis[(basis.m == mz)]
    # Prepare beam coefficients
    beamCoeff = focused_beam.C
    beamCoeff = beamCoeff[mz:]  # remove the first coefficient (j>=|l+p|)
    beamCoeff = np.repeat(beamCoeff, 2)  # 2 polarizations for each j
    beamCoeff = np.tile(beamCoeff, len(positions))  # repeat for each position
    
    
    # Prepare J values
    Jvals = np.repeat(np.arange(mz, maxJ+1), 2)
    Jvals = np.tile(Jvals, len(positions))  # repeat for each position
    # Compute total prefactor
    prefac = np.sqrt(2 * Jvals + 1) * (1j)**Jvals * beamCoeff
    # Define spherical wave (l=1, m=0) - will be modified
    spherical_wave = treams.spherical_wave(
        l=1, m=0, pol=1, k0=k, material=tm.material
    )
    
    # Expand the spherical wave in this basis to be able to change the coefficients
    coeffs = spherical_wave.expand(basis)
    # Ensure coefficients are set to prefac
    coeffs = coeffs * 0 + prefac
    # Filter the T-matrix for the specific m value
    tm_filtered = tm[:, (tm.basis.m == mz)]
    
    # Compute scattered field coefficients
    sca_coeffs = tm_filtered @ coeffs
    
    # Compute beam coefficients
    beam_coeffs = coeffs[:maxJ]
    
    return sca_coeffs, beam_coeffs, tm


def focusedBeam_treams_xs(focused_beam, sphere, position=None):
    """
    Compute scattering and extinction cross-sections for a focused beam interacting with a single sphere or a list of spheres.

    Parameters:
    -----------
    focused_beam : focusedBeam
        The focused beam object containing beam coefficients (wl, maxJ, mz, C)
    sphere : treams.TMatrix or list of treams.TMatrix
        Either a single Sphere TMatrix object or a list of Sphere TMatrix objects (for each k0)
    position : array_like, optional
        Position of the sphere. If None, assumes at origin

    Returns:
    --------
    sca : float or list
        Scattering cross-section(s) (in nm²)
    ext : float or list
        Extinction cross-section(s) (in nm²)
    """
    treams.config.POLTYPE = "parity"
    treams.config.MODETYPE = "regular"

    mz = focused_beam.mz
    maxJ = focused_beam.maxJ

    if position is None:
        position = [0, 0, 0]

    beamCoeff = focused_beam.C[mz:]
    beamCoeff_filtered = np.repeat(beamCoeff, 2)
    prefac = beamCoeff_filtered

    # Handle single sphere
    if not isinstance(sphere, list):
        wl = focused_beam.wl * 1000
        k0 = 2 * np.pi / wl
        sphere.k0 = k0
        basis = treams.SphericalWaveBasis.default(lmax=maxJ, nmax=1, positions=np.zeros((1, 3)))
        beamCoeff_full = np.zeros(len(basis), dtype=complex)
        indices = (basis.m == mz)
        beamCoeff_full[indices] = prefac
        spherical_wave = treams.spherical_wave(l=1, m=0, pol=1, k0=k0, material=sphere.material)
        coeffs = spherical_wave.expand(basis)
        coeffs = coeffs * 0 + beamCoeff_full
        coeffs.modetype = "regular"
        sca_val, ext_val = sphere.xs(coeffs)
        return sca_val, ext_val
    else:
        # List of spheres, each with its own k0
        sca = []
        ext = []
        basis = treams.SphericalWaveBasis.default(lmax=maxJ, nmax=1, positions=np.zeros_like(position))
        beamCoeff_full = np.zeros(len(basis), dtype=complex)
        indices = (basis.m == mz)
        beamCoeff_full[indices] = prefac
        for sph in sphere:
            k0 = sph.k0
            spherical_wave = treams.spherical_wave(l=1, m=0, pol=1, k0=k0, material=sph.material)
            coeffs = spherical_wave.expand(basis)
            coeffs = coeffs * 0 + beamCoeff_full
            coeffs.modetype = "regular"
            sca_val, ext_val = sph.xs(coeffs)
            sca.append(sca_val)
            ext.append(ext_val)
        return sca, ext

def focusedBeam_treams_xs_cluster(focused_beam, spheres, positions, k0s):
    """
    Compute scattering and extinction cross-sections for a focused beam interacting with a cluster of spheres.

    Parameters:
    -----------
    focused_beam : focusedBeam
        The focused beam object containing beam coefficients (wl, maxJ, mz, C)
    spheres : list of treams.TMatrix
        List of sphere TMatrix objects (for each particle radius)
    positions : array_like
        Positions of spheres. (N, 3)-dimensional
    k0s : array_like
        Array of wavenumbers

    Returns:
    --------
    sca : list
        Scattering cross-section(s) (in nm²)
    ext : list
        Extinction cross-section(s) (in nm²)
    """
    treams.config.POLTYPE = "parity"
    treams.config.MODETYPE = "regular"

    mz = focused_beam.mz
    maxJ = focused_beam.maxJ
    n_positions = len(positions)

    beamCoeff = focused_beam.C[mz:]
    beamCoeff_filtered = np.repeat(beamCoeff, 2)
    beamCoeff_filtered = np.tile(beamCoeff_filtered, n_positions)
    prefac = beamCoeff_filtered

    k0s = np.asarray(k0s)
    if n_positions != len(spheres):
        raise ValueError("Number of positions does not match number of spheres.")

    sca = []
    ext = []

    for k in k0s:
        for sphere in spheres:
            sphere.k0 = k
        materials = spheres[0].material

        tm = treams.TMatrix.cluster(spheres, positions).interaction.solve()
        basis = treams.SphericalWaveBasis.default(lmax=maxJ, nmax=n_positions, positions=np.zeros_like(positions))
        beamCoeff_full = np.zeros(len(basis), dtype=complex)
        indices = (basis.m == mz)
        beamCoeff_full[indices] = prefac
        spherical_wave = treams.spherical_wave(l=1, m=0, pol=1, k0=k, material=tm.material)
        coeffs = spherical_wave.expand(basis)
        coeffs = coeffs * 0 + beamCoeff_full
        coeffs.modetype = "regular"
        sca_val, ext_val = tm.xs(coeffs)
        sca.append(sca_val)
        ext.append(ext_val)
    return sca, ext
