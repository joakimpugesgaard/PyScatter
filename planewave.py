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
    def __init__(self, wavelength, domain, polarization = "x"):
        maxJ = 50
        self.wl = wavelength
        self.wn = 2 * np.pi / wavelength
        self.domain = domain
        self.size = domain.size
        self.polarization = polarization
        self.maxJ = maxJ
        self.poltypes = ["x", "y", "left circular", "right circular"]
        if polarization not in self.poltypes:
            raise ValueError(f"polarization must be one of {self.poltypes}")

        if polarization == "left circular":
            self.p = 1#np.array([1])
            self.super = 1
            self.prefac = 1
        elif polarization == "right circular":
            self.p = -1#np.array([-1])
            self.super = 1
            self.prefac = 1
        elif polarization == "x":
            self.p = np.array([1, -1])
            self.super = 1
            self.prefac = 1/np.sqrt(2)
            maxJ += 50
        elif polarization == "y":
            self.p = np.array([1, -1])
            self.super = -1
            self.prefac = 1/(np.sqrt(2)*1j)
            maxJ += 50
            
        self.mz_star = 1
        self.maxJ = maxJ
        # Initialize beam parameters dictionary
        self.beam_params = {
            'type': "planewave",
            'wl': self.wl,
            'mz_star': self.mz_star,
            'maxJ': self.maxJ,
         }

        super().__init__(maxJ, 1, wavelength, domain, nr=1, radius=0.25)
        self.planes = self.spherical_grids.keys()
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
    
    def compute_sum(self, spatial_fun = "bessel"):
        j0 = 1
        
        mp0L = self.get_multipoles(j0, 1, spatial_fun)
        mp0R = self.get_multipoles(j0, 1, spatial_fun)
        
        sumL = np.zeros_like(mp0L["magnetic"], dtype=complex)
        sumR = np.zeros_like(mp0L["magnetic"], dtype=complex)
        
        mp0L["magnetic"] *= 0
        mp0L["electric"] *= 0
        mp0R["magnetic"] *= 0
        mp0R["electric"] *= 0
        
  
        for j in range(j0, self.maxJ+1):
            mpp = self.get_multipoles(j, 1, spatial_fun)
            mpm = self.get_multipoles(j, -1, spatial_fun)
            
            mp0L["magnetic"] += (1j)**j * np.sqrt(2*j+1) * mpp["magnetic"]
            mp0L["electric"] += (1j)**j * np.sqrt(2*j+1) * mpp["electric"]
            mp0R["magnetic"] += (1j)**j * np.sqrt(2*j+1) * mpm["magnetic"]
            mp0R["electric"] += (1j)**j * np.sqrt(2*j+1) * mpm["electric"]
            
        sumL = sumL + np.sqrt(2*np.pi) * (mp0L["magnetic"]+(1j)*mp0L["electric"])
        sumR = sumR + np.sqrt(2*np.pi) * (mp0R["magnetic"]-(1j)*mp0R["electric"])
        
        if self.polarization == "left circular":
            sum = sumL
            
        elif self.polarization == "right circular":
            sum = sumR
        
        if self.polarization == "x":
            sum = (sumL + sumR) / np.sqrt(2)
            
        elif self.polarization == "y":
            sum = (sumL - sumR) / (np.sqrt(2)*1j)
        
        return sum 

    def plot_beam(self, plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: plot must be 'components' or 'total'
        """

        assert plot in ["components", "total"], "plot must be 'components' or 'total'"
    

        sum = self.compute_sum()
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)))
            sum[:] = np.real(sum[:]) 
                
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(np.abs(sum[:])) 
                vmax = np.max(np.abs(sum[:]))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
                
            fig.suptitle(f'Real parts of polarization components', fontsize=24, fontweight='bold')
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

            total_intensity = np.sum(np.abs(np.real(sum[:])**2) , axis=0)
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(r'Real part of total intensity', fontsize=24, fontweight='bold')
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
