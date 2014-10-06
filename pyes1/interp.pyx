
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil
import cython

ctypedef np.float64_t DOUBLE

def weight_cic(double[:] x,
               double[:] q,               
               int nx, double L,
               double rho0=0.):
    """ Weighting to grid (CIC)
    """

    cdef double dx = L/nx
    cdef int N  = x.shape[0]
    cdef np.ndarray[DOUBLE,ndim=1] grid = np.zeros(nx, dtype=np.float64)    
    cdef double xis
    cdef int i, left, right
    grid[:] = rho0
    for i in range(N):
        xis   = x[i]/dx
        left  = int(floor(xis))
        right = (left+1)%nx
        grid[left]  += q[i]*(left+1-xis)
        grid[right] += q[i]*(xis-left)

    return grid

def weight_S2(double[:] xp,
              double[:] q,
              int nx, double L):
    """ Weighting to grid (S2)
    """
    cdef int N = xp.shape[0]
    cdef np.ndarray[DOUBLE,ndim=1] rho = np.zeros(nx, dtype=np.float64)
    cdef double dx = L/nx
    cdef double xps, xs, qi
    for i in range(N):
        qi = q[i]
        xs = xp[i]/dx
        xps = int(round(xs))%nx
        xpsp = (xps+1)%nx
        xpsm = (xps-1)%nx

        rho[xps]  += qi*(3./4.-(xs-xps)**2)
        rho[xpsp] += qi*((1./2.-(xs-xps))**2)/2.
        rho[xpsm] += qi*((1./2.+(xs-xps))**2)/2.
        
    return rho
    
