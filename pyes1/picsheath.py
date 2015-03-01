
import numpy as np
from collections import namedtuple
import interp

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
def poisson_solve(b, dx, sigma):
    """ Assume V0=0
    """
    nx = len(b)
    A  = one_d_poisson(nx-1)
    A[-1,-1] = -1
    p  = -b*(dx**2)
    p[-1] = sigma*dx-p[-1]/2.
    x  = np.zeros_like(p)
    x[1:] = np.linalg.solve(A, p[1:])
    
    return x

# Dicts of weight/interp functions with string keys
__weight = {}
__interp = {}

__weight["CIC"] = interp.weight_cic_sheath

def interp_cic(E, xp, nx, L):
    """ Interpolate E to particle positions (CIC)
    """
    dx  = L/(nx-1.)
    xps = xp/dx # Scale
    left  = np.floor(xps).astype(np.int)
    right = np.mod(np.ceil(xps), nx).astype(np.int)
    E_interp = E[left]*(left+1-xps) + E[right]*(xps-left)
    
    return E_interp
__interp["CIC"] = interp_cic

def weight(xp, q, nx, L, method="CIC"):
    if method in __weight:
        dx = L/(nx-1.)
        return __weight[method](xp, q, nx, L)
    else:
        return method(xp, q, nx, L)

def interp(E, xp, nx, L, method="CIC"):
    if method in __interp:
        return __interp[method](E, xp, nx, L)
    else:
        return method(E, xp, nx, L)

def calc_E(phi, dx, sigma, E0=0):
    """ Calc E at the particle positions
    Centered difference (second order)
    """
    E       = np.zeros_like(phi)
    E[1:-1] = -(phi[2:]-phi[:-2])
    E[0]    = E0
    E[-1]   = -sigma
    
    return E/(2*dx)

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

def move(xp, vx, vy, dt, L, do_move=None):
    """ Move in place
    """
    if do_move is None:
        xp[:] = xp + dt*vx
    else:
        xp[do_move] = xp[do_move] + dt*vx[do_move]
    
def pic(species, nx, dx, nt, dt, L, B0, solver_method="FD", 
                                        weight_method="CIC",
                                        interp_method="CIC"):
    
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
    rhoa = np.zeros((nt+1, nx))
    siga = np.zeros(nt+1)

    # Main solution loop
    # Init half step back
    sigma = 0.
    rho = weight(xp, q, nx, L, method=weight_method)/dx
    phi = poisson_solve(rho, dx, sigma)
    E0  = calc_E(phi, dx, sigma)
    E   = interp(E0, xp, nx, L, method=interp_method)
    
    rotate(vx, vy, -wc, dt)
    accel(vx, vy, E, -qm, dt)

    xpa[0], vxa[0], vya[0]  = xp, vx, vy
    Ea[0], phia[0], rhoa[0] = E0, phi, rho

    sigma = 0.

    for i in range(1, nt+1):
        
        # Update velocity
        accel(vx, vy, E, qm, dt)
        rotate(vx, vy, wc, dt)
        accel(vx, vy, E, qm, dt)

        # Update position
        move(xp, vx, vy, dt, L, do_move=do_move)

        # Reflect particles
        reflect = xp < 0.
        xp[reflect] = -xp[reflect]
        vx[reflect] = -vx[reflect]
        vy[reflect] = -vy[reflect]

        # Update wall charge density
        hit = xp >= L
        sigma += np.sum(q[hit])
        xp[hit] = dx
        vx[hit] = 0.
        vy[hit] = 0.

        # Inject particles

        rho = weight(xp, q, nx, L, method=weight_method)/dx
        phi = poisson_solve(rho, dx, sigma)
        E0  = calc_E(phi, dx, sigma)
        E   = interp(E0, xp, nx, L, method=interp_method)
        
        xpa[i], vxa[i], vya[i]  = xp, vx, vy
        Ea[i], phia[i], rhoa[i] = E0, phi, rho
        siga[i] = sigma
    
    return (xpa, vxa, vya, Ea, phia, rhoa, siga)
