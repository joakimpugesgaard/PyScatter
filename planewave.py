import numpy as np
import scipy.special as sp
import scipy.integrate
from domain_class import domain
from Multipoles import Multipoles
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

class planeWave(Multipoles):
    def __init__(self, maxJ, wl, domain, polarization = "x"):
        super().__init__(maxJ, maxJ, wl, domain, nr=1, radius=0.25)
        self.wn = 2 * np.pi / wl
        self.domain = domain
        self.size = domain.size
        self.planes = self.spherical_grids.keys()
        self.maxJ = maxJ
        self.poltypes = ["x", "y", "left circular", "right circular"]
        if polarization not in self.poltypes:
            raise ValueError(f"polarization must be one of {self.poltypes}")

        self.wl = wl
        self.mz_star = 1
        
        # Initialize beam parameters dictionary
        self.beam_params = {
            'type': "planewave",
            'wl': self.wl,
            'mz_star': self.mz_star,
            'maxJ': self.maxJ,
            'nr': self.nr,
         }
        
        # initialize arrays of spherical coordinates. shape = (Nplanes, shape(grid))
        self.R = np.array([self.spherical_grids[plane][0] for plane in self.planes])
        self.Theta = np.array([self.spherical_grids[plane][1] for plane in self.planes])
        self.Phi = np.array([self.spherical_grids[plane][2] for plane in self.planes])
   
        
        #definepute beam coefficients
        self.C = np.ones(self.maxJ + 1, dtype=complex)

    
    def plot_Cjl(self):
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, self.maxJ + 1))
        color_idx = 0
        ax.bar(range(self.maxJ + 1), np.abs(self.C)**2, color=colors[color_idx], label=f"Polarization: {self.polarization}")
        ax.set_ylabel(r"$\mathbf{|C_{jm_zp}|}^2$", fontsize=30)
        ax.set_xlabel('j', fontsize=30)
        ax.set_xlim(0, self.maxJ)
        ax.set_xticks(range(0, self.maxJ + 1, 2))  # Set x-ticks to increase in steps of 2
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=20)
        fig.tight_layout()
        plt.show()
    
    def compute_sum(self, l, p, q, spatial_fun = "bessel"):
        
        m = l + p
        
        j0 = max(abs(m), 1)

        mp0 = self.get_multipoles(j0, m, spatial_fun)

        mp0["magnetic"] *= self.C[j0]*(1j)**j0 * np.sqrt(2*j0+1)
        mp0["electric"] *= self.C[j0]*(1j)**j0 * np.sqrt(2*j0+1)

        for j in range(j0+1, self.maxJ+1):
            mp = self.get_multipoles(j, m, spatial_fun)
            
            mp0["magnetic"] += (1j)**j * np.sqrt(2*j+1) * self.C[j] * mp["magnetic"]
            mp0["electric"] += (1j)**j * np.sqrt(2*j+1) * self.C[j] * mp["electric"]

        sum = mp0["magnetic"]+(1j)*p*mp0["electric"]

        return sum

    def plot_beam(self, l=None, p=None, q=None, interaction="scattering", plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            l (int): Orbital angular momentum quantum number.
            p (int): Azimuthal quantum number.
            q (int): Radial quantum number.
            maxJ (int): Maximum value of the total angular momentum quantum number.
            interaction (str, optional): The desired interaction to visualize. Can either be 'scattering' or 'internal', which will show the field respectively outside or inside 
                                a central region. Defaults to "scattering".
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: interaction must be 'scattering', 'internal' or 'both'
            ValueError: plot must be 'components' or 'total'
        """
        
        assert interaction in ["scattering", "internal", "both"], "interaction must be 'scattering' or 'internal'"
        assert plot in ["components", "total"], "plot must be 'components' or 'total'"
        
        rr = 0.25 * self.size  # Define the radius of the sphere
        
        # Initialize SCA and ABS arrays with ones, same shape as R
        SCA = np.ones_like(self.R)
        ABS = np.ones_like(self.R)
        
        # Find indices where values are less than or greater than rr
        index1 = np.where(self.R < rr)
        index2 = np.where(self.R > rr)

        # Set values in ABS and SCA arrays based on indices
        ABS[index2] = 0
        SCA[index1] = 0
        
        if interaction == "scattering":
            spatial_fun = "hankel"
            S = SCA
        elif interaction == "internal":
            spatial_fun = "bessel"
            S = ABS
        elif interaction == "both":
            spatial_fun = "bessel"
            S = np.ones_like(self.R)
        else:
            raise ValueError("interaction must be 'scattering' or 'internal'")
        
        if  (l is not None) or (p is not None) or (q is not None):
            if p not in [-1,1]:
                raise ValueError("p must be either -1 or 1")
            # If l, p, q are provided, set new C
            self.l = l
            self.p = p
            self.q = q
            self.C, self.lensInt, self.suma = self.C_jlp(l=l, p=p, q=q)

        sum = self.compute_sum(self.l, self.p, self.q, spatial_fun)
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)))
            
            sum[:] = np.abs(sum[:])**2 * S
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(np.abs(sum[:])) 
                vmax = np.max(np.abs(sum[:]))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
                
            fig.suptitle(f'Computed Sum (l={self.l}, p={self.p}, q={self.q})', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):

                im0 = axs[i, 0].imshow(np.abs(sum[0][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='hot', norm=norm)  
                axs[i, 0].set_title(r'$\xi_1$')
                axs[i, 0].set_xlabel(plane[0]+" [µm]")
                axs[i, 0].set_ylabel(plane[1]+" [µm]")
                axs[i, 0].tick_params(axis='both', which='both', direction='in')
                divider0 = make_axes_locatable(axs[i, 0])
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im0, cax=cax0)

                im1 = axs[i, 1].imshow(np.abs(sum[1][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='hot', norm=norm)
                axs[i, 1].set_title(r'$\xi_0$')
                axs[i, 1].set_xlabel(plane[0]+" [µm]")
                axs[i, 1].set_ylabel(plane[1]+" [µm]")
                axs[i, 1].tick_params(axis='both', which='both', direction='in')
                divider1 = make_axes_locatable(axs[i, 1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)

                im2 = axs[i, 2].imshow(np.abs(sum[2][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='hot', norm=norm)
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
            total_intensity = np.sum(np.abs(sum)**2 * S, axis=0)

            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(f'Total Intensity (l={self.l}, p={self.p}, q={self.q})', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):
                im = axs[i].imshow(total_intensity[i].T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='hot', norm=norm)
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
