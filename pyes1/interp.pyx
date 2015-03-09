
import numpy as np
cimport numpy as np
from libc.math cimport floor, ceil
import cython

ctypedef np.float64_t DOUBLE

def weight_cic(double[:] x,
               double[:] q,               
               int nx, double L,
               double rho0=0.):
    """ Weighting to periodic grid (CIC)
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

def weight_cic_sheath(double[:] x,
                      double[:] q,               
                      int nx, double L,
                      double rho0=0.):
    """ Weighting to non-periodic grid (CIC)
    """

    cdef double dx = L/(nx-1)
    cdef int N  = x.shape[0]
    cdef np.ndarray[DOUBLE,ndim=1] grid = np.zeros(nx, dtype=np.float64)    
    cdef double xis
    cdef int i, left, right
    grid[:] = rho0
    for i in range(N):
        xis   = x[i]/dx
        left  = int(floor(xis))
        right = left+1
        grid[left]  += q[i]*(left+1-xis)
        grid[right] += q[i]*(xis-left)

    return grid
    
def interp_cic_sheath(double[:] E,
                      double[:] xp,
                      int nx, double L):
    """ Interpolate E to particle positions (CIC)
    """
    cdef double dx  = L/(nx-1.)
    cdef double xps
    cdef double[:] E_interp = np.zeros_like(xp)
    cdef int i, left
    cdef int N = len(xp)

    for i in range(N):
        xps  = xp[i]/dx # Scale
        left = int(floor(xps))
        E_interp[i] = E[left]*(left+1-xps) + E[left+1]*(xps-left)
    
    return np.array(E_interp)
