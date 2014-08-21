
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

def poisson_solve_fd(b, nx, dx):
    """ Assume V0=0
    """
    A = one_d_poisson(nx-1)
    p = -b*(dx**2)
    x = np.zeros_like(p)
    x[1:] = np.linalg.solve(A, p[1:])
    
    return x

def poisson_solve_fft(rho):
    nx = len(rho)
    rhok = np.fft.fft(rho)
    k = nx*np.fft.fftfreq(nx)

    phik = np.zeros_like(rhok)
    phik[1:] = rhok[1:]/(k[1:]**2)
    sol = np.real(np.fft.ifft(phik))
    
    return sol

def poisson_solve(b, nx, dx, method="fd"):
    if method=="fd":
        return poisson_solve_fd(b, nx, dx)
    elif method=="fft":
        return poisson_solve_fft(b)
    else:
        return None

def weight(xp, q, nx, L):
    """ Weighting to grid (NGP)
    """
    rho = np.zeros(nx)
    xps = np.round(xp*nx/L).astype(np.int)
    xps[xps==nx] = 0
    for i in xrange(len(xps)):
        rho[xps[i]] += q[i]/epi0
        
    return rho

def interp(rho, E, xp, nx, L):
    """ Interpolate E to particle positions (NGP)
    """
    xps = np.round(xp*nx/L).astype(np.int)
    xps[xps==nx] = 0
    return E[xps]

def calc_E(phi, dx):
    """ Calc E at the particle positions
    The three 'bottom' steps on page 11
    """
    E   = np.zeros_like(phi)
    E[1:-1] = -(phi[2:]-phi[:-2])/(2*dx)
    E[0] = -(phi[1]-phi[-1])/(2*dx)
    E[-1] = -(phi[0]-phi[-1])/(2*dx)
    
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

def move(xp, vx, vy, dt, L):
    """ Move in place
    """
    xp[:] = xp + dt*vx
    
    xp[xp>=L] = xp[xp>=L] - L
    xp[xp<0]  = xp[xp<0]  + L

def pic(species, nx, dx, nt, dt, L, B0, method="fd"):
    
    N = 0
    for s in species: N += s.N

    q, qm, wc, xp, vx, vy = [np.zeros(N) for _ in range(6)]
    count = 0 # Trailing count
    for s in species:
        q[count:count+s.N]  = s.q
        qm[count:count+s.N] = s.q/s.m
        wc[count:count+s.N] = (s.q/s.m)*B0
        xp[count:count+s.N] = s.x0
        vx[count:count+s.N] = s.vx0
        vy[count:count+s.N] = s.vy0
        count += s.N

    # store the results at each time step
    xpa  = np.zeros((nt+1, N))
    vxa  = np.zeros((nt+1, N))
    vya  = np.zeros((nt+1, N))
    Ea   = np.zeros((nt+1, nx))
    phia = np.zeros((nt+1, nx))

    # Main solution loop
    # Init half step back
    rho = weight(xp, q, nx, L)
    phi = poisson_solve(rho, nx, dx, method=method)
    E0  = calc_E(phi, dx)
    E   = interp(rho, E0, xp, nx, L)
    
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
        move(xp, vx, vy, dt, L)

        rho = weight(xp, q, nx, L)
        phi = poisson_solve(rho, nx, dx, method=method)
        E0  = calc_E(phi, dx)
        E   = interp(rho, E0, xp, nx, L)
        
        xpa[i], vxa[i], vya[i] = xp, vx, vy
        Ea[i], phia[i] = E0, phi
    
    return (xpa, vxa, vya, Ea, phia)

    
