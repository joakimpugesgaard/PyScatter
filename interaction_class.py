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
    def __init__(self, maxJ, wl, domain, nr, radius, mu = 1, mu1 = 1, interaction = "both", beamtype = "focused"):
        self.l_max = maxJ
        self.m_max = maxJ*2+1
        super().__init__(maxJ, maxJ, wl, domain, nr)
        self.nr = nr
        self.rr = radius
        self.maxJ = maxJ
        self.wl = wl
        self.spherical_grids = domain.spherical_grid()
        self.size = domain.size
        self.planes = self.spherical_grids.keys()
        if beamtype == "focused":
            self.beam = focused_beams("LaguerreGauss", maxJ, wl, domain, p = 1, l = 3, q = 0, nr=1, NA=0.9, f=1000, n_lens=1)
            self.j0 = max(abs(self.beam.mz), 1)
            self.mz_star = self.beam.mz
            self.p = self.beam.p
            self.C = self.beam.C
        
        elif beamtype == "plane":
            self.C = np.ones((self.maxJ+1))
            self.j0 = 1
            self.p = 1
            
        elif beamtype == "displaced":
            self.beam_on = focused_beams("LaguerreGauss", self.maxJ, wl, domain, p = 1, l = 3, q = 0, nr=1, NA=0.9, f=1000, n_lens=1)
            self.mz_star = self.beam_on.mz
            self.p = self.beam_on.p
            self.C_on = self.beam_on.C
            
            self.beam_off = BeamDisplacement(domain, np.array([0,0.5,0]), wl, self.p, self.maxJ, NA=0.9, f=1000, n_lens=1)
            self.mz_star = self.beam_on.mz
            self.j0 = max(abs(self.mz_star), 1)
            self.Gjmp = self.beam_off.compute_C_off(self.mz_star, self.C_on)
            self.C = np.sum(self.Gjmp, axis=1)
        
            
        else:
            raise ValueError("Beam type not implemented")
        
        #test beamtype is correct
        if beamtype not in ['plane', 'gaussian', 'bessel', 'focused', 'displaced']:
            raise ValueError("beamtype must be either 'plane', 'gaussian', 'displaced', or 'bessel'")
        #test maxJ is correct
        if maxJ < 0:
            raise ValueError("maxJ must be greater than or equal to 0")
        #test wl is correct
        if wl <= 0:
            raise ValueError("wl must be greater than 0")
        self.k = 2*np.pi/self.wl
        self.x = self.k * self.rr
        
        self.mu = mu
        self.mu1 = mu1

            
        
        self.mie = glmt(self.maxJ, self.wl, self.nr, self.x, self.mu, self.mu1)


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
            S = SCA
        elif interaction == "internal":
            self.spatial_fun = "bessel"
            S = ABS
        elif interaction == "both":
            self.spatial_fun = "both"
            S = np.ones_like(self.R)
        else:
            raise ValueError("type must be 'scattering' or 'internal'")
    
    def compute_sum(self):
        #mie coefficients
        a = self.mie.a_j()
        b = self.mie.b_j()
        c = self.mie.c_j()
        d = self.mie.d_j()
        
        
        #initial multipole
        j0 = self.j0
        mp0 = self.get_multipoles(j0, self.mz_star, self.spatial_fun)
        
        mp0["magnetic"] *= self.C[j0]*(1j)**j0 * np.sqrt(2*j0+1) * (b[j0]*self.SCA + c[j0]*self.ABS)
        mp0["electric"] *= self.C[j0]*(1j)**j0 * np.sqrt(2*j0+1) * (a[j0]*self.SCA + d[j0]*self.ABS)

        for j in range(j0+1, self.maxJ):
            mp = self.get_multipoles(j, self.mz_star, self.spatial_fun)
            
            mp0["magnetic"] += (1j)**j * np.sqrt(2*j+1) * self.C[j] * (b[j]*self.SCA + c[j]*self.ABS) * mp["magnetic"] 
            mp0["electric"] += (1j)**j * np.sqrt(2*j+1) * self.C[j] * (a[j]*self.SCA + d[j]*self.ABS) * mp["electric"]

        sum = mp0["magnetic"]+(1j)* self.p * mp0["electric"]
        return sum
    
    def plot_int(self, plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: plot must be 'components' or 'total'
        """
        
        assert plot in ["components", "total"], "plot must be 'components' or 'total'"
        
        # Compute the sum of multipoles
        sum = self.compute_sum()
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)))
            
            sum[:] = np.abs(sum[:])**2
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
            total_intensity = np.sum(np.abs(sum)**2, axis=0)

            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(f'Total Intensity of Multipoles', fontsize=24, fontweight='bold')
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
