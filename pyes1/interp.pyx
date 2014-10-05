
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
