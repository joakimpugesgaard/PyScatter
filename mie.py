import numpy as np
import scipy.special as sp


class glmt:
    def __init__(self, maxJ, wl, nr, x, mu=1, mu1=1, dim = None):
        self.nr = nr
        self.maxJ = maxJ
        self.wl = wl
        self.x = x  # size parameter
        self.mu = mu
        self.mu1 = mu1
        self.dim = dim
        # test maxJ is correct
        if maxJ < 0:
            raise ValueError("maxJ must be greater than or equal to 0")
        # test wl is correct

        # Ensure no element in wl, nr, or R is negative
        if np.any(np.array(wl) <= 0):
            raise ValueError("All elements in wl must be greater than 0")
        if np.any(np.array(nr) <= 0):
            raise ValueError("All elements in nr must be greater than 0")
        if np.any(np.array(x) <= 0):
            raise ValueError("All elements in R must be greater than 0")
        
        self.k = 2 * np.pi / self.wl

        # If nr and wl are both 1D, and dimension is intended as 1D, reduce NR and X to coupled combinations
        if (self.dim == 1) and (self.nr.ndim == 1 and self.wl.ndim == 1 and self.nr.size == self.wl.size):
            self.NR = self.nr[:, np.newaxis]  # Make it a column vector
            self.X = self.x[:, np.newaxis]  # Make it a column vector
        elif self.dim == None:
            # prepare meshgrid of nr and R
            self.NR, self.X = np.meshgrid(self.nr, self.x)
        else:
            raise ValueError("Invalid dimensions specified. Use dim=1 for 1D (and make sure lengths of inputs are equal) or None for 2D.")
        
        # Precompute spherical Bessel and Hankel functions in a single loop for efficiency
        self.bx = []
        self.dbx = []
        self.bnr = []
        self.dbnr = []
        self.hx = []
        self.dhx = []
        for j in range(maxJ + 1):
            self.bx.append(sp.spherical_jn(j, self.X))
            self.dbx.append(sp.spherical_jn(j, self.X, True))
            self.bnr.append(sp.spherical_jn(j, self.NR * self.X))
            self.dbnr.append(sp.spherical_jn(j, self.NR * self.X, True))
            self.hx.append(self.hankel(j, self.X))
            self.dhx.append(self.hankel(j, self.X, True))

    def a_j(self):
        a = np.zeros((self.maxJ+1, np.shape(self.X)[0], np.shape(self.X)[1]), dtype=complex)
        for jj in range(self.maxJ+1):
            term1 = self.mu * self.NR**2 * self.bnr[jj] * (self.bx[jj] + self.X * self.dbx[jj])
            term2 = self.mu1 * self.bx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            term3 = self.mu * self.NR**2 * self.bnr[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term4 = self.mu1 * self.hx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            a[jj, :, :] = (term1 - term2) / (term3 - term4)
        return a.squeeze()

    def b_j(self):
        b = np.zeros((self.maxJ+1, np.shape(self.X)[0], np.shape(self.X)[1]), dtype=complex)
        for jj in range(self.maxJ+1):
            term1 = self.mu1 * self.bnr[jj] * (self.bx[jj] + self.X * self.dbx[jj])
            term2 = self.mu * self.bx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            term3 = self.mu1 * self.bnr[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term4 = self.mu * self.hx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            b[jj, :, :] = (term1 - term2) / (term3 - term4)
        return b.squeeze()

    def c_j(self):
        c = np.zeros((self.maxJ+1, np.shape(self.X)[0], np.shape(self.X)[1]), dtype=complex)
        for jj in range(self.maxJ+1):
            term1 = self.mu1 * self.bx[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term2 = self.mu1 * self.hx[jj] * (self.bx[jj] + self.X * self.dbx[jj])
            term3 = self.mu1 * self.bnr[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term4 = self.mu * self.hx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            c[jj, :, :] = (term1 - term2) / (term3 - term4)
        return c.squeeze()

    def d_j(self):
        d = np.zeros((self.maxJ+1, np.shape(self.X)[0], np.shape(self.X)[1]), dtype=complex)
        for jj in range(self.maxJ+1):
            term1 = self.mu1 * self.NR * self.bx[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term2 = self.mu1 * self.NR * self.hx[jj] * (self.bx[jj] + self.X * self.dbx[jj])
            term3 = self.mu * self.NR**2 * self.bnr[jj] * (self.hx[jj] + self.X * self.dhx[jj])
            term4 = self.mu1 * self.hx[jj] * (self.bnr[jj] + self.NR * self.X * self.dbnr[jj])
            d[jj, :, :] = (term1 - term2) / (term3 - term4)
        return d.squeeze()

    def hankel(self, n, x, derivative=False):
        # Returns the spherical hankel function of the first kind or its derivative
        if not derivative:
            return sp.spherical_jn(n, x) + 1j * sp.spherical_yn(n, x)
        else:
            return sp.spherical_jn(n, x, derivative=True) + 1j * sp.spherical_yn(n, x, derivative=True)
    
    
    