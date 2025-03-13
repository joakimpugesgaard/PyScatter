import numpy as np
import scipy
import scipy.special as sp
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from domain_class import domain
from Multipoles import Multipoles

class beamTypes:
    def __init__(self, domain, type):
        self.domain = domain
        self.type = type
        
    def LG(wl, q, l, w, z0, rho, z):
        """Returns a LaguerreGauss-polynomial with the parameters from the aplanatic lens model
        
        Args:
            wl (float): wavelength
            q (int): degree of the polynomial (radial index)
            l (float): azimuthal index
            w (float): beam waist
            z0 (float): Rayleigh range
            rho (float array): _description_
            z (float): z coordinate plane
        """
        wn = 2*np.pi/wl #wavenumber
        logN = lambda l,q: 0.5*(sp.gammaln(q+1)-sp.gammaln(q+l+1)-np.log(np.pi))

        bracket = 1j*(-(wn*rho**2*z)/(2*(z**2+z0)) + (2*q*l+1)*np.arctan(z/z0))
        
        #Calculate Laguerre polynomial
        l = abs(l)
        L = sp.genlaguerre(q, l)(2*rho**2/(w**2))

        logLG = logN(l,q) - rho**2/(w**2) + l*np.log(rho) + (l+1)*(0.5*np.log(2) - np.log(w))# + bracket 
        
        LG = np.exp(logLG)*L
        
        return LG