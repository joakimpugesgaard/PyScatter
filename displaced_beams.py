import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from focused_beams_class import *
from Multipoles import *

class BeamDisplacement:
    def __init__(self, domain, d = np.array([0, 1, 0]), wl=0.632, p=1, jmax=50, NA=0.9, f=1000, n_lens=1):
        self.wl = wl
        self.d = d
        self.p = p
        self.jmax = jmax
        self.k = 2 * np.pi / wl
        self.spherical_grids = domain.spherical_grid()
        self.size = domain.size
        self.planes = self.spherical_grids.keys()
        self.R = np.array([self.spherical_grids[plane][0] for plane in self.planes])
        self.focused_beam = focused_beams("LaguerreGauss", jmax, wl, domain, p=self.p, NA=NA, f=f, n_lens=n_lens)

        self.mpoles = Multipoles(l_max = jmax, m_max = jmax, wl = wl, domain = domain)
        self.C_on = self.focused_beam.C
        self.mz_star = self.focused_beam.mz
     
        self.CG1, self.CG2 = self.load_or_compute_CGs()
        
        self.mag, self.theta, self.phi = self.process_d(d)

        self.beam_params = {
            "type": "displaced",
            "d": d,
            "wl": wl,
            'mz_star': self.mz_star,
            "p": p,
            "maxJ": jmax,
            "NA": NA,
            "f": f,
            "n_lens": n_lens
        }

        
    def load_or_compute_CGs(self):
        try:
            data = np.load('CG1_CG2 copy.npz')
            CG1 = data['CG1']
            CG2 = data['CG2']
            print("Loaded CG1 and CG2 from file.")
        except FileNotFoundError:
            print("CG1 and CG2 not found. Computing...")
            CG1, CG2 = self.matrix_CGs(self.jmax, self.jmax)
            np.savez('CG1_CG2.npz', CG1=CG1, CG2=CG2)
        return CG1, CG2

    def matrix_CGs(self, Lmax, jpmax):
        CG1 = np.zeros((jpmax+1, Lmax+1, jpmax+1, 2*min(Lmax, jpmax)+1), dtype=np.complex128)
        CG2 = np.zeros((jpmax+1, Lmax+1, jpmax+1), dtype=np.complex128)

        for i in range(1, Lmax+1):
            for j in range(1, jpmax+1):
                for k in range(jpmax+1):
                    nnk = 0
                    for l in range(-min(i, j), min(i, j) + 1):
                        CG1[j, k, i, nnk] = self.clebsch_gordan(j, k, i, l, 0, l)
                        nnk += 1
                    CG2[j, k, i] = self.clebsch_gordan(j, k, i, 1, 0, 1)
        return CG1, CG2

    def process_d(self, d):
        if not isinstance(d, np.ndarray):
            d = np.array(d)
        if d.shape != (3,):
            raise ValueError("Input array must have shape (3,)")
        if np.linalg.norm(d) == 0:
            print("Displacement is zero")
            return 0, np.pi/2, 0
        mag = np.linalg.norm(d)
        d = d / mag
        theta = np.arccos(d[2])
        phi = np.arctan2(d[1], d[0])
        return mag, theta, phi

    def get_Jl(self, L, x):
        return sp.spherical_jn(L, x)

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
        
        if hyp == 0:
            d_jmp = 0
        else:
            d_jmp = np.exp(lnCoef) * cosFac * sinFac * hyp * (-1) ** (0.5 * (p - m - abs(m - p)))
        
        return d_jmp

    def D_jmp(self, j, m, p, alpha, beta, gamma):
        fac1 = np.exp(-1j * m * alpha)
        fac2 = self.d_jmp(j, m, p, beta)
        fac3 = np.exp(-1j * p * gamma)
        return fac1 * fac2 * fac3
    
    def clebsch_gordan(self, j1, j2, j, m1, m2, m):
        if any(2*x != int(2*x) for x in [j1, j2, j, m1, m2, m]):
            raise ValueError('All arguments must be integers or half-integers.')

        if m1 + m2 != m:
            return 0

        if not (j1 - m1 == int(j1 - m1)):
            return 0

        if not (j2 - m2 == int(j2 - m2)):
            return 0

        if not (j - m == int(j - m)):
            return 0

        if j > j1 + j2 or j < abs(j1 - j2):
            return 0

        if abs(m1) > j1:
            return 0

        if abs(m2) > j2:
            return 0

        if abs(m) > j:
            return 0
        m = float(m)
        cg = (-1)**(j1 - j2 + m) * np.sqrt(2 * j + 1) * self.wigner_3j([j1, j2, j], [m1, m2, -m])

        return cg
    
    def wigner_3j(self, j123, m123):
        j1, j2, j3 = j123
        m1, m2, m3 = m123

        if any(np.array(j123) < 0):
            raise ValueError('The j values must be non-negative')
        elif any(np.mod(np.array(j123 + m123), 0.5) != 0):
            raise ValueError('All arguments must be integers or half-integers')
        elif any(np.mod(np.array(j123) - np.array(m123), 1) != 0):
            raise ValueError('j123 and m123 do not match')

        if (j3 > (j1 + j2)) or (j3 < abs(j1 - j2)) or (m1 + m2 + m3 != 0) or any(np.abs(m123) > j123):
            return 0

        if all(m == 0 for m in m123) and np.mod(np.sum(j123), 2) != 0:
            return 0

        t1 = j2 - m1 - j3
        t2 = j1 + m2 - j3
        t3 = j1 + j2 - j3
        t4 = j1 - m1
        t5 = j2 + m2

        tmin = max(0, max(t1, t2))
        tmax = min(t3, min(t4, t5))

        t = np.arange(tmin, tmax + 1)
        
        gam1 = -np.ones(6) @ sp.gammaln(np.array([t, t - t1, t - t2, t3 - t, t4 - t, t5 - t]) + 1)
        gam2 = sp.gammaln(np.array([j1 + j2 + j3 + 1, j1 + j2 - j3, j1 - j2 + j3, -j1 + j2 + j3, 
                                        j1 + m1, j1 - m1, j2 + m2, j2 - m2, j3 + m3, j3 - m3]) + 1)
        
        gam2 = gam2 @ np.array([-1] + [1]*9)*0.5
        
        gamsum = np.squeeze(np.add.outer(gam1,gam2))
        
        w = np.sum((-1) ** t * np.exp(gamsum))* (-1) ** (j1 - j2 - m3)
        
        if np.isnan(w):
            print('Warning: Wigner3J is NaN!')
        elif np.isinf(w):
            print('Warning: Wigner3J is Inf!')
        return w
    
    def compute_C_off(self):
        """
        Compute C^off as a 2D array with:
        - j from 0 to j_max (axis 1)
        - m_z from -j_max to j_max (axis 2)

        Parameters:
        - j_max: Maximum j value
        - mz_star: Fixed m_z* value
        - k: Wavevector magnitude
        - d: Displacement vector (numpy array with shape (3,))
        - C_on: Known C^on values (1D array, indexed by j')

        Returns:
        - C_off: 2D array of shape (j_max+1, 2*j_max+1)
        """
        # Process displacement vector into spherical coordinates
        mag, theta, phi = self.process_d(self.d)

        # Initialize Lvals
        Lvals = np.arange(0, self.jmax)
        gjmp = np.zeros((self.jmax - self.mz_star + 1, self.jmax, 2 * self.jmax + 1), dtype=np.complex128)  # Initialize gjmp array
        
        # Use simpler expression for displacement along z
        if theta == 0:
            # Outer loop over j'
            for j_prime in range(max(abs(self.mz_star), 1), self.jmax + 1):
                for j in range(1, self.jmax + 1):  # Outer loop over j (start from 1 to match MATLAB)
                    n1 = 0
                    for m_z in range(j, -j - 1, -1):
                        Lfac = (2 * Lvals + 1) * (-1j)**Lvals * self.get_Jl(Lvals, self.k * mag)

                        # Compute gjmpi factor
                        gjmpi = (Lfac[:, None] * 
                                ((self.CG1[j, Lvals, j_prime, 0:2 * min(j, j_prime) + 1].conj().squeeze()) * self.CG2[j, Lvals, j_prime][:, None])
                            )  # Shape (2*mz+1, jmax+1)
            
                        # Compute Wigner matrix products (corresponding to AA and BB terms in MATLAB)
                        gjmp[j_prime - self.mz_star, j - 1, n1] = np.sum(np.conj(np.sum(gjmpi, axis=0).T), axis=0)

                        n1 += 1
        
        else:
            # Calculate the Wigner D matrices for each j
            D_matrices = []
            
            for jj in range(1, self.jmax + 1):
                size = 2 * jj + 1
                D_mat = np.zeros((size, size), dtype=complex)

                for m_idx, m in enumerate(range(-jj, jj + 1)):
                    for n_idx, n in enumerate(range(-jj, jj + 1)):
                        D_mat[m_idx, n_idx] = self.D_jmp(jj, m, n, phi, theta, 0)
                
                D_matrices.append(D_mat)  # Each element is a matrix of size (2*jj+1, 2*jj+1)
                
            printc = 0
            # Outer loop over j'
            for j_prime in range(max(abs(self.mz_star), 1), self.jmax + 1):
                printc += 1
                A = D_matrices[j_prime - 1]
                ii2 = j_prime - self.mz_star  # Adjust indexing
                for j in range(1, self.jmax + 1):  # Outer loop over j (start from 1 to match MATLAB)
                    B = D_matrices[j - 1]
                    n1 = 0

                    nn = np.arange(-min(j, j_prime), min(j, j_prime) + 1) 
                    for m_z in range(j, -j - 1, -1):
                        Lfac = (2 * Lvals + 1) * (-1j)**Lvals * self.get_Jl(Lvals, self.k * mag)

                        # Compute gjmpi factor
                        gjmpi = (Lfac[:, None] * 
                                ((self.CG1[j, Lvals, j_prime, 0:2 * min(j, j_prime) + 1].conj().squeeze()) * self.CG2[j, Lvals, j_prime][:, None])
                            )  # Shape (2*mz+1, jmax+1)
                        
                        
                        ii3 = j - m_z  # Adjust indexing
                        nn = np.arange(-min(j, j_prime), min(j, j_prime) + 1)  
                        DJ = A[ii2, j_prime - nn]  # Shape (2*jmax+1,)
                        DJp = B[ii3, j - nn] 

            
                        # Compute Wigner matrix products (corresponding to AA and BB terms in MATLAB)
                        gjmp[j_prime - self.mz_star, j - 1, n1] = np.sum((np.sum(gjmpi, axis=0).T).conj() * DJ.conj() * DJp, axis=0) 
                        n1 += 1
        
        Gjmp = np.sum((self.C_on.conj())[1:, None, None] * gjmp[:, :, :], axis=0)  # Shape (jmax, 2*jmax+1)
        return Gjmp

    def C_off_to_mx(self, C_off_mz = None):
        if C_off_mz is None:
            C_off_mz = self.compute_C_off()
        else:
            C_off_mz = C_off_mz
        C_off_mx = np.zeros_like(C_off_mz, dtype=np.complex128)
        D_matrices = []

        for jj in range(1, np.shape(C_off_mz)[0]):
            size = 2 * jj + 1
            D_mat = np.zeros((size, size), dtype=complex)
            for m_idx, m in enumerate(range(-jj, jj + 1)):
                for n_idx, n in enumerate(range(-jj, jj + 1)):
                    D_mat[m_idx, n_idx] = self.d_jmp(jj, m, n, np.pi / 2)
            D_matrices.append(D_mat)

        for jj in range(1, np.shape(C_off_mz)[0]):
            D_mat = D_matrices[jj - 1]
            cx = 0
            for m_x in range(jj, -jj - 1, -1):
                C_off_mx[jj - 1, cx] = np.sum(C_off_mz[jj - 1, :2 * jj] * D_mat[cx, :2 * jj])
                cx += 1
        return C_off_mx
    
    def C_on_to_mx(self):
        C_on_mx = np.zeros((self.jmax + 1, 2 * self.jmax + 1), dtype=np.complex128)
        for j in range(self.jmax + 1):
            m_vals = np.arange(-j, j + 1)
            for idx, mx in enumerate(m_vals):
                d = self.d_jmp(j, mx, self.p, -np.pi / 2)
                C_on_mx[j, self.jmax + mx] = d * self.C_on[j]
        return C_on_mx

    def plot_stacked_histograms(self, C_off = None, basis = "mz", title="Stacked Histograms"):
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        if C_off is None:
            if basis == "mz":
                C_off = self.compute_C_off()
                C_on = self.C_on[:, np.newaxis]
            elif basis == "mx":
                C_off = self.C_off_to_mx()
                C_on = self.C_on_to_mx()
        else:
            C_off = C_off
            if basis == "mz":
                C_on = self.C_on[:, np.newaxis]
            elif basis == "mx":
                C_on = self.C_on_to_mx()

        C_off_transposed = C_off.T

        # Define a unique color for each value in the histogram using the tab20 colormap
        # Generate 51 distinct colors with minimal similarity between neighbors
        colors = plt.cm.tab20(np.linspace(0, 1, 2* self.jmax+10))
        np.random.seed(35)  
        np.random.shuffle(colors)

        ax[0].bar(
            np.arange(1, C_off_transposed.shape[1] + 1),
            np.abs(C_off_transposed[0])**2,
            color=colors[0]
        )
        for i in range(1, C_off_transposed.shape[0]):
            ax[0].bar(
            np.arange(1, C_off_transposed.shape[1] + 1),
            np.abs(C_off_transposed[i])**2,
            bottom=np.sum(np.abs(C_off_transposed[:i])**2, axis=0),
            color=colors[i]
            )

        ax[0].set_title(f"C_off Stacked Histogram in basis {basis}", fontsize=15)
        ax[0].set_xlabel('j', fontsize=15)
        ax[0].set_ylabel(r"$\mathbf{|C^{off}_{jm_xp}|}^2$", fontsize=15)

        C_on_mx_stacked = np.array([np.abs(C_on[j])**2 for j in range(len(C_on))]).T
        
        colors_on_mx = plt.cm.tab20(np.linspace(0, 1, C_on_mx_stacked.shape[0]))

        ax[1].bar(
            np.arange(1, C_on_mx_stacked.shape[1] + 1),
            C_on_mx_stacked[0],
            color=colors_on_mx[0]
        )
        for i in range(1, C_on_mx_stacked.shape[0]):
            ax[1].bar(
                np.arange(1, C_on_mx_stacked.shape[1] + 1),
                C_on_mx_stacked[i],
                bottom=np.sum(C_on_mx_stacked[:i], axis=0),
                color=colors_on_mx[i]
            )

        ax[1].set_title(f"C_on Stacked Histogram in basis {basis}", fontsize=15)
        ax[1].set_xlabel('j', fontsize=15)
        ax[1].set_ylabel(r"$\mathbf{|C^{on}_{jm_zp}|}^2$", fontsize=15)

        fig.suptitle(title, fontsize=18)
        fig.tight_layout()
        plt.show()
    
    def compute_sum(self, C_off=None, spatial_fun="bessel"):
        if C_off is None:
            Gjmp = self.compute_C_off()
        else:
            Gjmp = C_off

        j0 = max(abs(self.mz_star), 1)

        mp0 = self.mpoles.get_multipoles(j0, 0, spatial_fun)

        mp0["magnetic"] *= 0
        mp0["electric"] *= 0

        for j in range(j0, np.shape(Gjmp)[0]):
            nn = 0
            for m in range(j, -j - 1, -1):
                mp = self.mpoles.get_multipoles(j, m, spatial_fun)

                mp0["magnetic"] += (1j) ** j * np.sqrt(2 * j + 1) * Gjmp[j-1, nn] * mp["magnetic"]
                mp0["electric"] += (1j) ** j * np.sqrt(2 * j + 1) * Gjmp[j-1, nn] * mp["electric"]
                nn += 1
        sum = mp0["magnetic"] + (1j) * self.p * mp0["electric"]
    
        return sum
        
    def plot_beam(self, sum = None, interaction="scattering", plot="components", globalnorm=False):
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
        if sum is None:
            spatial_fun = "bessel"
            S = np.ones_like(self.R)
            sum = self.compute_sum(spatial_fun=spatial_fun)
        else:
            sum = sum
        
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
                
            fig.suptitle(f'Intensity d = {self.mag} µm', fontsize=24, fontweight='bold')
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

            fig.suptitle(f'Total Intensity (d = {self.mag} µm)', fontsize=24, fontweight='bold')
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
