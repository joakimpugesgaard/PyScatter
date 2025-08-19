import numpy as np
import scipy.special as sp
from scipy.integrate import trapezoid
from src.domain import domain
from src.multipoles import multipoles
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmcrameri.cm as cmc

class focusedBeam(multipoles):
    def __init__(self, maxJ, wavelength, domain, p = 1, l = 0, q = 0, nr=1, NA=0.9, f=1000, n_lens = 1):
        super().__init__(maxJ, maxJ, wavelength, domain, nr)
        self.NA = NA
        self.f = f
        self.wn = 2 * np.pi / wavelength
        self.domain = domain
        self.size = domain.size
        self.planes = self.spherical_grids.keys()
        self.n_lens = n_lens
        self.maxJ = maxJ
        if p not in [-1,1]:
            raise ValueError("Helicity must be either -1 or 1")
        self.p = p
        self.l = l
        self.q = q
        self.nr = nr
        self.mz = l + p
        self.wl = wavelength
        self.polarization = "left circular" if p == 1 else "right circular"
        # Initialize beam parameters dictionary
        self.beam_params = {
            'type': "focused",
            'q': self.q,
            'l': self.l,
            'p': self.p,
            'wl': self.wl,
            'mz_star': self.mz,
            'polarization': self.polarization,
            'maxJ': self.maxJ,
            'nr': self.nr,
            'NA': self.NA,
            'f': self.f,
            'n_lens': self.n_lens
        }
        

        
        self.beam = self.LaguerreGauss
        
        #compute beam coefficients
        self.C, self.lensInt, self.suma = self.beamCoeffs()
        #print("The (2j+1)Cjm_z normalization yields %.6f" % self.suma)
        #print("The LG integral on the aplanatic lens surface is %.3f\n" % self.lensInt)

        
    def d_jmp(self, j, m, p, Theta):
        M = max(abs(m), abs(p))
        N = min(abs(m), abs(p))
        
        lnCoef = 0.5 * (sp.gammaln(j - M + 1) + sp.gammaln(j + M + 1) - sp.gammaln(j + N + 1) - sp.gammaln(j - N + 1))

        cosFac = np.cos(Theta / 2) ** (abs(m + p))
        sinFac = (np.sin(Theta / 2)) ** (abs(m - p))
        
        n = j - M
        alpha = abs(m - p)
        beta = abs(m + p)
        hyp = sp.eval_jacobi(n, alpha, beta, np.cos(Theta))
        
        if np.all(hyp == 0):
            d_jmp = 0
        else:
            d_jmp = np.exp(lnCoef) * cosFac * sinFac * hyp * (-1) ** (0.5 * (p - m - abs(m - p)))
        
        return d_jmp

    def beamCoeffs(self, l = None, p = None, q = None):

        if l is None:
            l = self.l
        if p is None:
            p = self.p
        if q is None:
            q = self.q
            
        if p not in [-1,1]:
            raise ValueError("p must be either -1 or 1")
        
        mz = l + p
        
        assert self.maxJ >= abs(l)+p, "maxJ must be greater than or equal to abs(l)+p"
        
        theta_max = np.arcsin(self.NA / self.n_lens)  # maximum half angle
        theta = np.linspace(0.0001, theta_max, 250)
        
        # Generalize beam calculation
        beam_params = {
            'q': q,
            'l': l,
            'rho': self.f * np.sin(theta),
            'z': self.f
        }
        
        beamfac = self.beam(**beam_params)
        
        lensInt = trapezoid(self.f**2 * 2 * np.pi * np.sin(theta) * np.cos(theta) * (beamfac**2), theta)
        
        C = np.zeros(self.maxJ + 1, dtype=complex)
        suma = 0
        for j in range(max(abs(mz), 1), self.maxJ+1):
            C[j] = trapezoid(
                np.sin(theta) *
                self.f * np.exp(-1j * self.wn * self.f) *
                np.sqrt(2 * np.pi) *
                np.sqrt(self.n_lens * np.cos(theta)) *
                beamfac *
                self.d_jmp(j, mz, p, theta), theta
            )
            suma += np.abs(C[j])**2 * (2 * j + 1)
        
        return C, lensInt, suma
    
    def plotBeamCoeffs(self, l=None, p=None, q=None):
        if l is None:
            l = [self.l]
        if p is None:
            p = [self.p]
        if q is None:
            q = [self.q]
        # Ensure l, p, and q are lists
        if not isinstance(l, list):
            l = [l]
        if not isinstance(p, list):
            p = [p]
        if not isinstance(q, list):
            q = [q]
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(l) * len(p) * len(q)))
        color_idx = 0

        for l_val in l:
            for p_val in p:
                for q_val in q:
                    # Compute C for each combination
                    C, lensInt, suma = self.beamCoeffs(l=l_val, p=p_val, q=q_val)
                    label = f"mz={l_val + p_val}, NA={self.NA:.1f}, l={l_val}, p={p_val}, q={q_val}"
                    ax.bar(np.arange(0, len(C)), np.abs(C)**2, 
                    label=label, alpha=0.6)#, color=colors[color_idx])
                    color_idx += 1

        ax.set_ylabel(r"$\mathbf{|C_{jm_zp}|}^2$", fontsize=30)
        ax.set_xlabel('j', fontsize=30)
        ax.set_xlim(0, self.maxJ)
        if self.maxJ <= 30:
            ax.set_xticks(range(0, self.maxJ + 1, 2))  # Set x-ticks to increase in steps of 2
        else:
            ax.set_xticks(range(0, self.maxJ + 1, 5))  # Set x-ticks to increase in steps of 5
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=20)

        fig.tight_layout()
        plt.show()
    
    def LaguerreGauss(self, q, l, rho, z, **kwargs):
        wn = 2 * np.pi / self.wl  # wavenumber
        w = self.get_w(self.NA, abs(l))

        z0 = self.n_lens * w / self.NA  # Rayleigh distance
        logN = lambda l, q: 0.5 * (sp.gammaln(q + 1) - sp.gammaln(q + l + 1) - np.log(np.pi))
        bracket = 1j * (-(wn * rho**2 * z) / (2 * (z**2 + z0)) + (2 * q * l + 1) * np.arctan(z / z0))
        
        l = abs(l)
        L = sp.genlaguerre(q, l)(2 * rho**2 / (w**2))
        logLG = logN(l, q) - rho**2 / (w**2) + l * np.log(rho) + (l + 1) * (0.5 * np.log(2) - np.log(w))
        LG = np.exp(logLG) * L
        
        return LG
    
    def get_w(self, NA, l):
        w_values = {
            0.25: [140, 121, 108, 98, 90, 84, 79, 75, 72],
            0.3: [170, 144, 130, 121, 113, 106, 101, 97, 94],
            0.4: [220, 182, 170, 169, 150, 142, 136, 130, 125],
            0.5: [270, 236, 216, 200, 186, 178, 170, 162, 155],
            0.6: [320, 280, 255, 236, 222, 210, 201, 193, 185],
            0.7: [395, 340, 300, 281, 265, 250, 237, 226, 216],
            0.8: [440, 370, 320, 295, 281, 268, 259, 249, 240],
            0.9: [500, 420, 390, 360, 335, 290, 317, 303, 290]
        }   
        
        if NA not in w_values:
            raise ValueError("NA value not recognized. Available values are 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9.")
        
        if abs(l) >= len(w_values[NA]):
            raise ValueError("l value out of range (maximum |l| value is 8).")
        
        return w_values[NA][l]
    
    
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

    def plotBeam(self, plot="components", globalnorm=False):
        """Plot the computed sum of multipoles.

        Args:
            plot (str, optional): Plot full intensity or each polarization component. Defaults to "components".
            globalnorm (bool, optional): Normalize each plot to its own max (see individual behavior) or global max (see which components dominate). Defaults to False.

        Raises:
            ValueError: interaction must be 'scattering', 'internal' or 'both'
            ValueError: plot must be 'components' or 'total'
        """
        
        assert plot in ["components", "total"], "plot must be 'components' or 'total'"

        sum = self.compute_sum(self.l, self.p, self.q, "bessel")
        
        if plot == "components":
            # Plot Nself.planes x 3 subplots
            fig, axs = plt.subplots(len(self.planes), 3, figsize=(12, 4 * len(self.planes)), squeeze = False)
            
            sum[:] = np.abs(sum[:])**2 
            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(np.abs(sum[:])) 
                vmax = np.max(np.abs(sum[:]))
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None
                
            fig.suptitle(f'Computed Sum (l={self.l}, p={self.p}, q={self.q})', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):

                im0 = axs[i, 0].imshow(np.abs(sum[0][i]).T, extent=(-0.5*self.size, 0.5*self.size, -0.5*self.size, 0.5*self.size), origin='lower', cmap='cmc.batlow', norm=norm)  
                axs[i, 0].set_title(r'$\xi_1$')
                axs[i, 0].set_xlabel(plane[0]+" [µm]")
                axs[i, 0].set_ylabel(plane[1]+" [µm]")
                axs[i, 0].tick_params(axis='both', which='both', direction='in')
                divider0 = make_axes_locatable(axs[i, 0])
                cax0 = divider0.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im0, cax=cax0)

                im1 = axs[i, 1].imshow(np.abs(sum[1][i]).T, extent=(-0.5*self.size, 0.5*self.size, -0.5*self.size, 0.5*self.size), origin='lower', cmap='cmc.batlow', norm=norm)
                axs[i, 1].set_title(r'$\xi_0$')
                axs[i, 1].set_xlabel(plane[0]+" [µm]")
                axs[i, 1].set_ylabel(plane[1]+" [µm]")
                axs[i, 1].tick_params(axis='both', which='both', direction='in')
                divider1 = make_axes_locatable(axs[i, 1])
                cax1 = divider1.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im1, cax=cax1)

                im2 = axs[i, 2].imshow(np.abs(sum[2][i]).T, extent=(-0.5*self.size, 0.5*self.size, -0.5*self.size, 0.5*self.size), origin='lower', cmap='cmc.batlow', norm=norm)
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
            fig, axs = plt.subplots(len(self.planes), 1, figsize=(12, 4 * len(self.planes)), squeeze=False)

            # Sum the intensities of the three components
            total_intensity = np.sum(np.abs(sum)**2, axis=0)

            if globalnorm:
                # Find the global min and max values for normalization
                vmin = np.min(total_intensity)
                vmax = np.max(total_intensity)
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            else:
                norm = None

            fig.suptitle(f'Total Intensity (l={self.l}, p={self.p}, q={self.q})', fontsize=24, fontweight='bold')
            for i, plane in enumerate(self.planes):
                # Handle both single and multiple plane cases
                ax = axs[i, 0] if len(self.planes) > 1 else axs[0, 0]
                im = ax.imshow(total_intensity[i].T, extent=(-0.5*self.size, 0.5*self.size, -0.5*self.size, 0.5*self.size), origin='lower', cmap='cmc.batlow', norm=norm)
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
