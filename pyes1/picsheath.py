
import numpy as np
from collections import namedtuple
from interp import weight_cic_sheath as weight

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
    p[-1] = -sigma*dx+p[-1]/2.
    x  = np.zeros_like(p)
    x[1:] = np.linalg.solve(A, p[1:])
    
    return x

def interp(E, xp, nx, L):
    """ Interpolate E to particle positions (CIC)
    """
    dx  = L/(nx-1.)
    xps = xp/dx # Scale
    left  = np.floor(xps).astype(np.int)
    right = np.mod(np.ceil(xps), nx).astype(np.int)
    E_interp = E[left]*(left+1-xps) + E[right]*(xps-left)
    
    return E_interp

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
    
def pic(electron, ion, nx, dx, nt, dt, L, B0):
    
    N = 0
    for s in [electron, ion]: N += s.N
    N_max = N*3

    q, qm, wc, xp, vx, vy = [np.zeros(N_max) for _ in range(6)]
    el      = np.ndarray((N_max,), dtype=np.bool)
    do_move = np.ndarray((N_max,), dtype=np.bool)
    do_move[:] = False
    count = 0 # Trailing count
    for s,t in [(electron,True), (ion,False)]:
        q[count:count+s.N]  = s.q
        qm[count:count+s.N] = s.q/s.m
        wc[count:count+s.N] = (s.q/s.m)*B0
        xp[count:count+s.N] = s.x0
        vx[count:count+s.N] = s.vx0
        vy[count:count+s.N] = s.vy0
        el[count:count+s.N] = t
        do_move[count:count+s.N] = s.m>0
        count += s.N

    # store the results at each time step
    xpa  = np.zeros((nt+1, N_max))
    vxa  = np.zeros((nt+1, N_max))
    vya  = np.zeros((nt+1, N_max))
    Ea   = np.zeros((nt+1, nx))
    phia = np.zeros((nt+1, nx))
    rhoa = np.zeros((nt+1, nx))
    siga = np.zeros(nt+1)
    Na   = np.zeros(nt+1)
    ela  = np.zeros((nt+1,N_max), dtype=np.bool)

    # Main solution loop
    # Init half step back
    sigma = 0.
    rho = weight(xp[:N], q[:N], nx, L)/dx
    phi = poisson_solve(rho, dx, sigma)
    E0  = calc_E(phi, dx, sigma)
    E   = interp(E0, xp[:N], nx, L)
    
    rotate(vx[:N], vy[:N], -wc[:N], dt)
    accel(vx[:N], vy[:N], E, -qm[:N], dt)

    xpa[0], vxa[0], vya[0]  = xp, vx, vy
    Ea[0], phia[0], rhoa[0] = E0, phi, rho
    ela[0] = el
    Na[0] = N
    sigma = 0.

    for i in range(1, nt+1):
        
        # Update velocity
        accel(vx[:N], vy[:N], E[:N], qm[:N], dt)
        #rotate(vx[:N], vy[:N], wc, dt)
        accel(vx[:N], vy[:N], E[:N], qm[:N], dt)

        # Update position
        move(xp[:N], vx[:N], vy[:N], dt, L)

        # Reflect particles
        reflect = xp[:N] < 0.
        xp[:N][reflect] = -xp[:N][reflect]
        vx[:N][reflect] = -vx[:N][reflect]
        vy[:N][reflect] = -vy[:N][reflect]

        # Update wall charge density
        hit = xp[:N] >= L
        n_hit = np.sum(hit)
        sigma += np.sum(q[:N][hit])

        # Remove particles that hit the wall
        xp[:N-n_hit] = xp[:N][~hit].copy()
        vx[:N-n_hit] = vx[:N][~hit].copy()
        vy[:N-n_hit] = vy[:N][~hit].copy()
        q[:N-n_hit]  = q[:N][~hit].copy()
        qm[:N-n_hit] = qm[:N][~hit].copy()
        el[:N-n_hit] = el[:N][~hit].copy()
        N -= n_hit

        # Add particles from source term

        n_pairs = 20
        vxe = np.random.choice(electron.vx0, size=n_pairs)
        vxi = np.random.choice(ion.vx0, size=n_pairs)
        xe_new = np.random.rand(n_pairs)*20.0*dx
        xi_new = np.random.rand(n_pairs)*20.0*dx

        xp[N:N+n_pairs] = xe_new
        vx[N:N+n_pairs] = vxe
        vy[N:N+n_pairs] = 0.0
        q[N:N+n_pairs]  = electron.q
        qm[N:N+n_pairs] = electron.q/electron.m
        el[N:N+n_pairs] = True
        N += n_pairs
        
        xp[N:N+n_pairs] = xi_new
        vx[N:N+n_pairs] = vxi
        vy[N:N+n_pairs] = 0.0
        q[N:N+n_pairs]  = ion.q
        qm[N:N+n_pairs] = ion.q/ion.m
        el[N:N+n_pairs] = False
        N += n_pairs

        # Calc fields
        rho = weight(xp[:N], q[:N], nx, L)/dx
        phi = poisson_solve(rho, dx, sigma)
        E0  = calc_E(phi, dx, sigma)
        E   = interp(E0, xp[:N], nx, L)
        
        xpa[i], vxa[i], vya[i]  = xp, vx, vy
        Ea[i], phia[i], rhoa[i] = E0, phi, rho
        siga[i], ela[i], Na[i]  = sigma, el, N
    
    return (xpa, vxa, vya, Ea, phia, rhoa, siga, ela, Na)
