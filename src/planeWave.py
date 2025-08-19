import numpy as np
from src.domain import domain
from src.multipoles import multipoles
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmcrameri.cm as cmc

class planeWave(multipoles):
    def __init__(self, wavelength, domain, polarization = "x"):
        maxJ = 40
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

        elif polarization == "y":
            self.p = np.array([1, -1])
            self.super = -1
            self.prefac = 1/(np.sqrt(2)*1j)

            
        self.mz_star = 1
        # Initialize beam parameters dictionary
        self.beam_params = {
            'type': "planewave",
            'wl': self.wl,
            'mz_star': self.mz_star,
            'maxJ': self.maxJ,
            'polarization': self.polarization,
         }

        super().__init__(maxJ, 1, wavelength, domain, nr=1)
        self.planes = self.spherical_grids.keys()
        # initialize arrays of spherical coordinates. shape = (Nplanes, shape(grid))
        self.R = np.array([self.spherical_grids[plane][0] for plane in self.planes])
        self.Theta = np.array([self.spherical_grids[plane][1] for plane in self.planes])
        self.Phi = np.array([self.spherical_grids[plane][2] for plane in self.planes])
        
        cart_coords = domain.cart_coords()
        self.X = np.array([cart_coords[plane][0] for plane in self.planes])
        self.Y = np.array([cart_coords[plane][1] for plane in self.planes])
        self.Z = np.array([cart_coords[plane][2] for plane in self.planes])
        
        #definepute beam coefficients
        self.C = np.ones(self.maxJ+1, dtype=complex)
        self.C[0] = 0
    
    def compute_sum(self, spatial_fun = "bessel"):
        j0 = 1
        
        mp0L = self.get_multipoles(j0, 1, spatial_fun, nr = 1)
        mp0R = self.get_multipoles(j0, 1, spatial_fun, nr = 1)
        
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
    
    def compute_pol(self):
        
        field = np.zeros((3, len(self.planes), self.R[0].shape[0], self.R[0].shape[1]), dtype=complex)
        
        if self.polarization == "left circular":
            for i in range(len(self.planes)):
                # xi_1
                field[0][i] = np.exp(1j * self.wn * self.Z[i])
            labels = [r"$\xi_1$", r"$\xi_0$", r"$\xi_{-1}$"]

        elif self.polarization == "right circular":
            for i in range(len(self.planes)):
                # xi_-1
                field[2][i] = np.exp(1j * self.wn * self.Z[i])
            labels = [r"$\xi_1$", r"$\xi_0$", r"$\xi_{-1}$"]
            
        elif self.polarization == "x":
            for i in range(len(self.planes)):
                # x
                field[0][i] = np.exp(1j * self.wn * self.Z[i])
            labels = [r"$e_x$", r"$e_y$", r"$e_z$"]
        
        elif self.polarization == "y":
            for i in range(len(self.planes)):
                # y
                field[1][i] = np.exp(1j * self.wn * self.Z[i])
            labels = [r"$e_x$", r"$e_y$", r"$e_z$"]
            
        return field, labels
 
    
    
    
    def plotBeam(self, plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: plot must be 'components' or 'total'
        """

        assert plot in ["components", "total"], "plot must be 'components' or 'total'"


        #sum = self.compute_sum()
        sum, labels = self.compute_pol()
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)), squeeze = False)
            sum[:] = np.real(sum[:]) 
            
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(sum[:]) 
                vmax = np.max(sum[:])
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
                
            fig.suptitle(f'Real parts of polarization components', fontsize=24, fontweight='bold')
            axs[0, 0].set_title(labels[0])
            axs[0, 1].set_title(labels[1])
            axs[0, 2].set_title(labels[2])
            for i, plane in enumerate(self.planes):

                im0 = axs[i, 0].imshow(np.abs(sum[0][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='cmc.batlow', norm=norm)             
                axs[i, 0].set_xlabel(plane[0]+" [µm]")
                axs[i, 0].set_ylabel(plane[1]+" [µm]")
                axs[i, 0].tick_params(axis='both', which='both', direction='in')
                divider0 = make_axes_locatable(axs[i, 0])
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im0, cax=cax0)

                im1 = axs[i, 1].imshow(np.abs(sum[1][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='cmc.batlow', norm=norm)
                axs[i, 1].set_xlabel(plane[0]+" [µm]")
                axs[i, 1].set_ylabel(plane[1]+" [µm]")
                axs[i, 1].tick_params(axis='both', which='both', direction='in')
                divider1 = make_axes_locatable(axs[i, 1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)

                im2 = axs[i, 2].imshow(np.abs(sum[2][i]).T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='cmc.batlow', norm=norm)
                axs[i, 2].set_xlabel(plane[0]+" [µm]")
                axs[i, 2].set_ylabel(plane[1]+" [µm]")
                axs[i, 2].tick_params(axis='both', which='both', direction='in')
                divider2 = make_axes_locatable(axs[i, 2])
                cax2 = divider2.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im2, cax=cax2)
            fig.subplots_adjust(hspace=-0.8, wspace=-0.3)
            fig.tight_layout()
            plt.show()

        elif plot == "total":
            # Plot self.planes x 1 subplots with the summed intensity of the three components
            fig, axs = plt.subplots(len(self.planes), 1, figsize=(12, 4 * len(self.planes)), squeeze = False)

            total_intensity = np.sum(np.abs(np.real(sum[:])) , axis=0)
            #total_intensity = np.abs(np.real(np.sum(sum, axis = 0)))
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(r'Real part of total intensity', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):
                ax = axs[i, 0] if len(self.planes) > 1 else axs[0, 0]
                im = ax.imshow(total_intensity[i].T, extent=(-self.size, self.size, -self.size, self.size), origin='lower', cmap='cmc.batlow', norm=norm)
                ax.set_title(f'{plane} plane')
                ax.set_xlabel(plane[0] + " [µm]")
                ax.set_ylabel(plane[1] + " [µm]")
                ax.tick_params(axis='both', which='both', direction='in')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            fig.subplots_adjust(hspace=-0.8, wspace=-0.2)
            fig.tight_layout()
            plt.show()
