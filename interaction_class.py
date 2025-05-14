import numpy as np
import matplotlib
import scipy
import scipy.special as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import rc
from focused_beams_class import *
from domain_class import *
from displaced_beams import *
from mie import *
from Multipoles import *
import numba
import scipy
import numba_scipy

## Define Standard Units
fsize = 22
tsize = 15
tdir = 'in'
major = 5
minor = 3
style = 'default'

params = {
    'figure.figsize': (15,12),
    'savefig.dpi': 75,
    'text.usetex': False,
    'font.size': fsize,
    'legend.fontsize': tsize,
    'legend.title_fontsize': tsize,
    'mathtext.fontset' : 'stix',
    'font.family' : 'STIXGeneral',    
    'axes.labelsize':15,
    'axes.titlesize':20,
    'lines.linewidth':2.5,
    'axes.grid': False,
    'axes.labelweight':'bold',
    'legend.loc': 'upper right',
    'xtick.labelsize':'x-small',
    'ytick.labelsize':'x-small',
}
plt.rcParams.update(params)

class interaction(Multipoles):
    def __init__(self, beam, domain, nr, radius, mu = 1, mu1 = 1, interaction = "both"):
        self.beam_params = beam.beam_params
        self.beamtype = self.beam_params["type"]
        self.maxJ = self.beam_params["maxJ"]
        self.wl = self.beam_params["wl"]
        self.l_max = self.maxJ
        self.m_max = self.maxJ
        self.rr = radius
        super().__init__(self.l_max, self.m_max, self.wl, domain, nr=nr, radius=radius)
        self.nr = nr
        self.spherical_grids = domain.spherical_grid()
        self.size = domain.size
        self.planes = self.spherical_grids.keys()
        
        self.mz_star = self.beam_params["mz_star"]
        self.j0 = max(abs(self.mz_star), 1)
        self.p = self.beam_params["p"]

        if self.beamtype == "plane":
            self.C = np.ones((self.maxJ+1, 3))
        elif self.beamtype == "focused":
            self.C = beam.C
        elif self.beamtype == "displaced":
            self.Gjmp  = beam.compute_C_off()
            self.C = self.Gjmp            
        else:
            raise ValueError("Beam type not implemented")
        
        self.k = 2*np.pi/self.wl
        self.x = self.k * self.rr
        
        self.mu = mu
        self.mu1 = mu1

        
        self.mie = glmt(self.maxJ, self.wl, self.nr, self.x, self.mu, self.mu1)
        
        #mie coefficients
        self.a = self.mie.a_j()
        self.b = self.mie.b_j()
        self.c = self.mie.c_j()
        self.d = self.mie.d_j()

        # Initialize SCA and ABS arrays with ones, same shape as R
        SCA = np.ones_like(self.R)
        ABS = np.ones_like(self.R)
        
        # Find indices where values are less than or greater than rr
        index1 = np.where(self.R < self.rr)
        index2 = np.where(self.R > self.rr)

        # Set values in ABS and SCA arrays based on indices
        ABS[index2] = 0
        SCA[index1] = 0
        
        if interaction == "scattering":
            self.spatial_fun = "hankel"
            self.S = SCA
        elif interaction == "internal":
            self.spatial_fun = "bessel"
            self.S = ABS
        elif interaction == "both":
            self.spatial_fun = "both"
            self.S = np.ones_like(self.R)
        else:
            raise ValueError("type must be 'scattering' or 'internal'")
        
    def _is_equal(self, a, b):
            """
            Helper function to check equality between two inputs.
            Returns True if:
            1. Both are arrays and are equal.
            2. Both are scalars and are equal.
            Returns False if:
            1. Arrays are unequal.
            2. Scalars are unequal.
            3. One is an array and the other is a scalar.
            """
            if np.isscalar(a) and np.isscalar(b):
                return a == b
            elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                return np.array_equal(a, b)
            else:
                return False
            
    def compute_sum(self):
        assert self.a.ndim == 1, "a must be 1D"
        #initial multipole
        j0 = self.j0

        

        if (self.beamtype == "plane") or (self.beamtype == "focused"):
            Js = np.arange(self.maxJ+1)
            coefs = 1j**Js * np.sqrt(2*Js+1)
            prefac = (self.C.T * coefs).T

            mp0 = self.get_multipoles(j0, self.mz_star, self.spatial_fun)
            
            mp0["magnetic"] *= prefac[j0] * (self.b[j0] * self.SCA + self.c[j0] * self.ABS)
            mp0["electric"] *= prefac[j0] *(self.a[j0] * self.SCA + self.d[j0] * self.ABS)
    
            for j in range(j0+1, self.maxJ):
                mp = self.get_multipoles(j, self.mz_star, self.spatial_fun)
                
                mp0["magnetic"] += prefac[j] *(self.b[j] * self.SCA + self.c[j] * self.ABS)* mp["magnetic"] 
                mp0["electric"] += prefac[j] *(self.a[j] * self.SCA + self.d[j] * self.ABS)* mp["electric"]

            sum = mp0["magnetic"]+(1j)* self.p * mp0["electric"]
            
        elif self.beamtype == "displaced":
            Js = np.arange(j0, self.maxJ+1)
            coefs = 1j**Js * np.sqrt(2*Js+1)
            prefac = (self.C.T * coefs).T
            
            mp0 = self.get_multipoles(j0, self.mz_star, self.spatial_fun)
            
            mp0["magnetic"] *= 0
            mp0["electric"] *= 0

            for j in range(j0, self.maxJ):
                nn = 0
                a = self.a[j]
                b = self.b[j]
                c = self.c[j]
                d = self.d[j]
                for m in range(-j, j + 1):
                    mp = self.get_multipoles(j, m, self.spatial_fun)

                    mp0["magnetic"] += prefac[j, nn] *(b * self.SCA + c * self.ABS)* mp["magnetic"]
                    mp0["electric"] += prefac[j, nn] *(a * self.SCA + d * self.ABS)* mp["electric"]
                    nn += 1
                    
            sum = mp0["magnetic"] + (1j) * self.p * mp0["electric"]
            
            
        return sum
    
    def getCrossSection(self, type="scattering", **kwargs):
        """Compute the cross section of the multipoles.
        
        Args:
            type (str, optional): Type of cross section to compute. Defaults to "scattering".
            **kwargs: Optional parameters to override x, radius, nr, or wl.

        Raises:
            ValueError: type must be 'scattering', 'internal' or 'extinction'
        
        Returns: 
            Array of cross sections for the specified type.
        """
        # Allow overriding of x, radius, nr, wl via direct arguments 
        x = kwargs.pop("x", getattr(self, "x", None))
        radius = kwargs.pop("radius", getattr(self, "rr", None))
        nr = kwargs.pop("nr", getattr(self, "nr", None))
        wl = kwargs.pop("wl", getattr(self, "wl", None))
 
        # Use self.mie if parameters match, else create new glmt instance
        if not (self._is_equal(x, getattr(self, "x", None)) and
            self._is_equal(nr, getattr(self, "nr", None)) and
            self._is_equal(wl, getattr(self, "wl", None)) and
            self._is_equal(radius, getattr(self, "rr", None))):
            k = 2 * np.pi / wl
            x = k * radius
            
            mie = glmt(self.maxJ, wl, np.asarray(nr), x)
            #calculate mie coefficients
            a = mie.a_j()
            b = mie.b_j()
            c = mie.c_j()
            d = mie.d_j()
        else:
            mie = self.mie
            a = self.a
            b = self.b
            c = self.c
            d = self.d

        if type == "scattering":
            #ensure C is 2D to make matrix multiplication work
            if self.C.ndim == 1:
                C = self.C[:, None]
            L = np.arange(0, len(C))
            coefs = np.array([1j**L * np.sqrt(2*L+1)])
            
            prefac = (0.5*np.abs(C.T * coefs)**2).T
            
            # Expand prefac to match the shape of a for broadcasting
            while prefac.ndim -1 < a.ndim:
                prefac = np.expand_dims(prefac, axis=-1)
                
            Ws = prefac * (np.abs(a)**2 + np.abs(b)**2) # [J, mZ, x, nr]
            Ws = np.sum(Ws, axis=(0, 1)) #sum over J and mZ
            
            return Ws
    
        elif type == "internal":
            #ensure C is 2D to make matrix multiplication work
            if self.C.ndim == 1:
                C = self.C[:, None]
            L = np.arange(0, len(C))
            coefs = np.array([1j**L * np.sqrt(2*L+1)])
            
            prefac = 0.5*np.abs(C.T * coefs)**2
            
            # Expand prefac to match the shape of a for broadcasting
            while prefac.ndim -1 < c.ndim:
                prefac = np.expand_dims(prefac, axis=-1)
            
            Wint = prefac * (np.abs(c)**2 + np.abs(d)**2) # [J, mZ, x, nr]
            Wint = np.sum(Wint, axis=(0, 1)) #sum over J and mZ
            
            return Wint
        elif type == "extinction":
            #ensure C is 2D to make matrix multiplication work
            if self.C.ndim == 1:
                C = self.C[:, None]
            L = np.arange(0, len(C))
            coefs = np.array([1j**L * np.sqrt(2*L+1)])
            
            prefac = 0.5*np.abs(C.T * coefs)**2
            
            # Expand prefac to match the shape of a for broadcasting
            while prefac.ndim -1 < c.ndim:
                prefac = np.expand_dims(prefac, axis=-1)
            
            Wext = prefac * (a + b) # [J, mZ, x, nr]
            Wext = 2*np.real(np.sum(Wext, axis=(0, 1))) #sum over J and mZ
            
            return Wext
        
        else:
            raise ValueError("type must be 'scattering', 'internal' or 'extinction'")
        
    def plotCrossSection(self, type="scattering", radius=None, nr=None):
        if np.any(radius == None):
            radius = self.rr
        if np.any(nr == None):
            nr = self.nr
        C = self.getCrossSection(type=type, radius=radius, nr=nr)
        
        # Print info
        if C.ndim == 0:
            print("C has unexpected number of dimensions: 0")
            print(f"Radius: {radius if radius is not None else self.rr}, nr: {nr if nr is not None else self.nr}, C_{type}: {C}")
        # 1D plot
        if C.ndim == 1:
            # Pick x and label
            if radius is not None and getattr(radius, "ndim", 0) and radius.size > 1:
                x, xlabel = radius, "Radius"
                title = "nr: " + str(nr)
            elif nr is not None and getattr(nr, "ndim", 0) and nr.size > 1:
                x, xlabel = nr, "Refractive Index (nr)"
                title = "Radius: " + str(radius)
            elif getattr(self.rr, "ndim", 0) and self.rr.size > 1:
                x, xlabel = self.rr, "Radius"
                title = "nr: " + str(nr)
            elif getattr(self.nr, "ndim", 0) and self.nr.size > 1:
                x, xlabel = self.nr, "Refractive Index (nr)"
                title = "Radius: " + str(radius)
            else:
                x, xlabel = np.arange(len(C)), "Index"
            plt.figure(figsize=(8, 5))
            plt.plot(x, C, linewidth=2, color='blue')
            plt.xlabel(xlabel)
            plt.ylabel(f"C_{type}")
            plt.title(title)
            plt.grid(True)
            plt.show()
        # 2D plot
        elif C.ndim == 2:
            if radius is not None and nr is not None:
                extent = [nr.min(), nr.max(), radius.min(), radius.max()]
                xlabel, ylabel = "Refractive Index (nr)", "Radius"
            elif radius is not None:
                extent = [0, C.shape[1], radius.min(), radius.max()]
                xlabel, ylabel = "Index", "Radius"
            elif nr is not None:
                extent = [nr.min(), nr.max(), 0, C.shape[0]]
                xlabel, ylabel = "Refractive Index (nr)", "Index"
            else:
                extent = None
                xlabel, ylabel = "Index", "Index"
            plt.figure(figsize=(8, 6))
            plt.imshow(C, aspect='auto', origin='lower', extent=extent, cmap='viridis')
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f"Cross Section ({type})")
            plt.colorbar(label=f"C_{type}")
            plt.show()
        else:
            print("C has unexpected number of dimensions:", C.ndim)
    
    def plot_int(self, radius = None, nr = None, plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: plot must be 'components' or 'total'
        """
        
        assert plot in ["components", "total"], "plot must be 'components' or 'total'"
        if radius is None and nr is None:
            # Check if self.rr or self.nr are arrays with more than one element, raise error if so
            if (isinstance(self.rr, np.ndarray) and self.rr.size > 1) or (isinstance(self.nr, np.ndarray) and self.nr.size > 1):
                raise ValueError("self.rr and self.nr must be scalars or arrays with a single element, not arrays with more than one element.")
        elif radius is not None:
            # Check if radius is an array with more than one element, raise error if so
            if isinstance(radius, np.ndarray) and (radius.size > 1):
                raise ValueError("radius must be a scalar or an array with a single element.")
            elif isinstance(self.nr, np.ndarray) and (self.nr.size > 1):
                raise ValueError("self.nr must be a scalar or an array with a single element.")
        elif nr is not None:
            # Check if nr is an array with more than one element, raise error if so
            if isinstance(nr, np.ndarray) and nr.size > 1:
                raise ValueError("nr must be a scalar or an array with a single element.")
            elif isinstance(self.rr, np.ndarray) and (self.rr.size > 1):
                raise ValueError("self.rr must be a scalar or an array with a single element.")
        
        # If radius or nr are provided and differ from self.rr or self.nr, re-initialize the superclass
        if (radius is not None and not self._is_equal(radius, self.rr)) or (nr is not None and not self._is_equal(nr, self.nr)):
            super().__init__(self.l_max, self.m_max, self.wl, self.domain, nr=nr if nr is not None else self.nr, radius=radius if radius is not None else self.rr)
            
        
        # Compute the sum of multipoles
        sum = self.compute_sum()
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)))
            
            sum[:] = np.abs(sum[:])**2
            sum *= self.S
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(np.abs(sum[:])) 
                vmax = np.max(np.abs(sum[:]))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
                
            fig.suptitle(f'Computed Sum of Multipoles', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):

                im0 = axs[i, 0].imshow(np.abs(sum[0][i]).T, extent=(-self.rr, self.rr, -self.rr, self.rr), origin='lower', cmap='hot', norm=norm)  
                axs[i, 0].set_title(r'$\xi_1$')
                axs[i, 0].set_xlabel(plane[0]+" [µm]")
                axs[i, 0].set_ylabel(plane[1]+" [µm]")
                axs[i, 0].tick_params(axis='both', which='both', direction='in')
                divider0 = make_axes_locatable(axs[i, 0])
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im0, cax=cax0)

                im1 = axs[i, 1].imshow(np.abs(sum[1][i]).T, extent=(-self.rr, self.rr, -self.rr, self.rr), origin='lower', cmap='hot', norm=norm)
                axs[i, 1].set_title(r'$\xi_0$')
                axs[i, 1].set_xlabel(plane[0]+" [µm]")
                axs[i, 1].set_ylabel(plane[1]+" [µm]")
                axs[i, 1].tick_params(axis='both', which='both', direction='in')
                divider1 = make_axes_locatable(axs[i, 1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)

                im2 = axs[i, 2].imshow(np.abs(sum[2][i]).T, extent=(-self.rr, self.rr, -self.rr, self.rr), origin='lower', cmap='hot', norm=norm)
                axs[i, 2].set_title(r'$\xi_{-1}$')
                axs[i, 2].set_xlabel(plane[0]+" [µm]")
                axs[i, 2].set_ylabel(plane[1]+" [µm]")
                axs[i, 2].tick_params(axis='both', which='both', direction='in')
                divider2 = make_axes_locatable(axs[i, 2])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax2)
            fig.subplots_adjust(hspace=-0.8, wspace=-0.2)
            fig.tight_layout()
            plt.show()
        
        elif plot == "total":
            # Plot self.planes x 1 subplots with the summed intensity of the three components
            fig, axs = plt.subplots(len(self.planes), 1, figsize=(12, 4 * len(self.planes)))

            # Sum the intensities of the three components
            total_intensity = np.sum(np.abs(sum*self.S)**2, axis=0)

            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(f'Total Intensity', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):
                im = axs[i].imshow(total_intensity[i].T, extent=(-self.rr, self.rr, -self.rr, self.rr), origin='lower', cmap='hot', norm=norm)
                axs[i].set_title(f'{plane} plane')
                axs[i].set_xlabel(plane[0] + " [µm]")
                axs[i].set_ylabel(plane[1] + " [µm]")
                axs[i].tick_params(axis='both', which='both', direction='in')
                divider = make_axes_locatable(axs[i])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            fig.subplots_adjust(hspace=-0.8, wspace=-0.2)
            fig.tight_layout()
            plt.show()
