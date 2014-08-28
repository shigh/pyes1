
import numpy as np
from collections import namedtuple

epi0 = 1. # Epsilon0
Species = namedtuple("Species", ["q", "m", "N", "x0", "vx0", "vy0"])

__cache_one_d_poisson = {}
def one_d_poisson(n):
    if n in __cache_one_d_poisson:
        return __cache_one_d_poisson[n]
    
    a = np.zeros((n,n))
    np.fill_diagonal(a, -2.)
    np.fill_diagonal(a[:-1,1:], 1.)
    np.fill_diagonal(a[1:,:-1], 1.)
    __cache_one_d_poisson[n] = a
    
    return  a

# Dict of solver functions with string keys
__solver = {}    
def poisson_solve_fd(b, dx):
    """ Assume V0=0
    """
    nx = len(b)
    A  = one_d_poisson(nx-1)
    p  = -b*(dx**2)
    x  = np.zeros_like(p)
    x[1:] = np.linalg.solve(A, p[1:])
    
    return x
__solver["FD"] = poisson_solve_fd

def poisson_solve_fft(rho, dx):
    nx   = len(rho)
    rhok = np.fft.fft(rho)
    k    = np.fft.fftfreq(nx)*2*np.pi/dx

    phik     = np.zeros_like(rhok)
    phik[1:] = rhok[1:]/(k[1:]**2)
    sol      = np.real(np.fft.ifft(phik))
    
    return sol
__solver["FFT"] = poisson_solve_fft

def poisson_solve(b, dx, method="FD"):
    if method in __solver:
        return __solver[method](b, dx)
    else:
        return method(b, dx)

# Dicts of weight/interp functions with string keys
__weight = {}
__interp = {}

def weight_cic(xp, q, nx, L):
    """ Weighting to grid (CIC)
    """
    rho = np.zeros(nx)
    # Scale from [0,L] to [0,nx]
    xps   = xp*nx/L
    left  = np.floor(xps).astype(np.int)
    right = np.mod(np.ceil(xps), nx).astype(np.int)
    for i in xrange(len(xps)):
        rho[left[i]]  += q[i]*(left[i]+1-xps[i])
        rho[right[i]] += q[i]*(xps[i]-left[i])
        
    return rho
__weight["CIC"] = weight_cic
    
def interp_cic(E, xp, nx, L):
    """ Interpolate E to particle positions (CIC)
    """
    xps = xp*nx/L
    left  = np.floor(xps).astype(np.int)
    right = np.mod(np.ceil(xps), nx).astype(np.int)
    E_interp = E[left]*(left+1-xps) + E[right]*(xps-left)
    
    return E_interp
__interp["CIC"] = interp_cic
    
def weight_ngp(xp, q, nx, L):
    """ Weighting to grid (NGP)
    """
    rho = np.zeros(nx)
    xps = np.round(xp*nx/L).astype(np.int)
    xps[xps==nx] = 0
    for i in xrange(len(xps)):
        rho[xps[i]] += q[i]
        
    return rho
__weight["NGP"] = weight_ngp

def interp_ngp(E, xp, nx, L):
    """ Interpolate E to particle positions (NGP)
    """
    xps = np.round(xp*nx/L).astype(np.int)
    xps[xps==nx] = 0
    return E[xps]
__interp["NGP"] = interp_ngp

def weight(xp, q, nx, L, method="NGP"):
    if method in __weight:
        return __weight[method](xp, q, nx, L)
    else:
        return method(xp, q, nx, L)

def interp(E, xp, nx, L, method="NGP"):
    if method in __interp:
        return __interp[method](E, xp, nx, L)
    else:
        return method(E, xp, nx, L)

def calc_E(phi, dx):
    """ Calc E at the particle positions
    The three 'bottom' steps on page 11
    """
    E       = np.zeros_like(phi)
    E[1:-1] = -(phi[2:]-phi[:-2])/(2*dx)
    E[0]    = -(phi[1]-phi[-1])/(2*dx)
    E[-1]   = -(phi[0]-phi[-1])/(2*dx)
    
    return E

def accel(vx, vy, E, alpha, dt):
    """ Accel in place
    """
    vx[:] = vx + alpha*E*dt/2.
    # vy is unchanged

def rotate(vx, vy, wc, dt):
    """ Rotate in place
    """
    c = np.cos(wc*dt)
    s = np.sin(wc*dt)
    vx_new =  c*vx + s*vy
    vy_new = -s*vx + c*vy

    vx[:] = vx_new
    vy[:] = vy_new

def normalize(x, L):
    """ Keep x in [0,L), assuming a periodic domain
    """
    # The order here is significant because of rounding
    # If x<0 is very close to 0, then float(x+L)=L
    x[x<0]  = x[x<0]  + L
    x[x>=L] = x[x>=L] - L
    
def move(xp, vx, vy, dt, L, do_move=None):
    """ Move in place
    """
    if do_move==None:
        xp[:] = xp + dt*vx
    else:
        xp[do_move] = xp[do_move] + dt*vx[do_move]

    normalize(xp, L)

def pic(species, nx, dx, nt, dt, L, B0, solver_method="FD", 
                                        weight_method="NGP",
                                        interp_method="NGP"):
    
    N = 0
    for s in species: N += s.N

    q, qm, wc, xp, vx, vy = [np.zeros(N) for _ in range(6)]
    do_move = np.ndarray((N,), dtype=np.bool)
    do_move[:] = False
    count = 0 # Trailing count
    for s in species:
        q[count:count+s.N]  = s.q
        qm[count:count+s.N] = s.q/s.m
        wc[count:count+s.N] = (s.q/s.m)*B0
        xp[count:count+s.N] = s.x0
        vx[count:count+s.N] = s.vx0
        vy[count:count+s.N] = s.vy0
        do_move[count:count+s.N] = s.m>0
        count += s.N

    # store the results at each time step
    xpa  = np.zeros((nt+1, N))
    vxa  = np.zeros((nt+1, N))
    vya  = np.zeros((nt+1, N))
    Ea   = np.zeros((nt+1, nx))
    phia = np.zeros((nt+1, nx))

    # Main solution loop
    # Init half step back
    rho = weight(xp, q, nx, L, method=weight_method)
    phi = poisson_solve(rho/epi0, dx, method=solver_method)
    E0  = calc_E(phi, dx)
    E   = interp(E0, xp, nx, L, method=interp_method)
    
    rotate(vx, vy, -wc, dt)
    accel(vx, vy, E, -qm, dt)

    xpa[0], vxa[0], vya[0] = xp, vx, vy
    Ea[0], phia[0] = E0, phi
    
    for i in range(1, nt+1):
        
        # Update velocity
        accel(vx, vy, E, qm, dt)
        rotate(vx, vy, wc, dt)
        accel(vx, vy, E, qm, dt)

        # Update position
        move(xp, vx, vy, dt, L, do_move=do_move)
      
        rho = weight(xp, q, nx, L, method=weight_method)
        phi = poisson_solve(rho/epi0, dx, method=solver_method)
        E0  = calc_E(phi, dx)
        E   = interp(E0, xp, nx, L, method=interp_method)
        
        xpa[i], vxa[i], vya[i] = xp, vx, vy
        Ea[i], phia[i] = E0, phi
    
    return (xpa, vxa, vya, Ea, phia)

    
