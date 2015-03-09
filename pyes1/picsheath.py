
import numpy as np
import scipy as sp
import scipy.sparse as sps
import scipy.sparse.linalg
from collections import namedtuple
from interp import weight_cic_sheath as weight,\
                   interp_cic_sheath as interp

Species = namedtuple("Species", ["q", "m", "N", "x0",
                                 "vx0", "vy0", "sample",
                                 "source"])

__cache_one_d_poisson = {}
def one_d_poisson(n):
    if n in __cache_one_d_poisson:
        return __cache_one_d_poisson[n]
    
    a = np.zeros((n,n))
    np.fill_diagonal(a, -2.)
    np.fill_diagonal(a[:-1,1:], 1.)
    np.fill_diagonal(a[1:,:-1], 1.)
    a = sps.csr_matrix(a)
    __cache_one_d_poisson[n] = a
    
    return  a

# Dict of solver functions with string keys
def poisson_solve(b, dx, sigma):
    """ Assume V0=0
    """
    nx = len(b)
    A  = one_d_poisson(nx-1)
    A[-1,-1] = -1
    p  = -b*(dx**2)
    p[-1] = -sigma*dx+p[-1]/2.
    x  = np.zeros_like(p)
    x[1:] = sps.linalg.spsolve(A, p[1:])
    
    return x

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
    
def pic(electron, ion, nx, dx, nt, dt, L, B0, save_res=None,
        n_pairs=2, L_source=1.0):
    
    N = 0
    for s in [electron, ion]: N += s.N
    N_max = N*2

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
    save_xp  = 'xp'  in save_res
    save_vx  = 'vx'  in save_res
    save_vy  = 'vy'  in save_res
    save_E   = 'E'   in save_res
    save_phi = 'phi' in save_res
    save_rho = 'rho' in save_res
    save_sig = 'sig' in save_res
    save_N   = 'N'   in save_res
    save_el  = 'el'  in save_res
    if save_xp or save_vx or save_vy:
        save_N = save_el = True

    if save_xp:  xpa  = np.zeros((nt+1, N_max))
    if save_vx:  vxa  = np.zeros((nt+1, N_max))
    if save_vy:  vya  = np.zeros((nt+1, N_max))
    if save_E:   Ea   = np.zeros((nt+1, nx))
    if save_phi: phia = np.zeros((nt+1, nx))
    if save_rho: rhoa = np.zeros((nt+1, nx))
    if save_sig: siga = np.zeros(nt+1)
    if save_N:   Na   = np.zeros(nt+1)
    if save_el:  ela  = np.zeros((nt+1,N_max), dtype=np.bool)

    # Main solution loop
    # Init half step back
    sigma = 0.
    rho = weight(xp[:N], q[:N], nx, L)/dx
    phi = poisson_solve(rho, dx, sigma)
    E0  = calc_E(phi, dx, sigma)
    E   = interp(E0, xp[:N], nx, L)
    
    #rotate(vx[:N], vy[:N], -wc[:N], dt)
    accel(vx[:N], vy[:N], E, -qm[:N], dt)

    if save_xp:  xpa[0]  = xp
    if save_vx:  vxa[0]  = vx
    if save_vy:  vya[0]  = vy
    if save_E:   Ea[0]   = E0
    if save_phi: phia[0] = phi
    if save_rho: rhoa[0] = rho
    if save_sig: siga[0] = sigma
    if save_N:   Na[0]   = N
    if save_el:  ela[0]  = el

    for i in range(1, nt+1):
        
        # Update velocity
        accel(vx[:N], vy[:N], E[:N], qm[:N], dt)
        #rotate(vx[:N], vy[:N], wc, dt)
        accel(vx[:N], vy[:N], E[:N], qm[:N], dt)

        # Update position
        move(xp[:N], vx[:N], vy[:N], dt, L)

        # Reflect particles
        reflect = xp[:N] < 0.
        n_reflect  = np.sum(reflect)
        ne_reflect = np.sum(el[:N][reflect])
        ni_reflect = n_reflect-ne_reflect
        xp[:N][reflect] = 0.0
        vx[:N][  el[:N] &reflect] = np.abs(electron.sample(ne_reflect))
        vx[:N][(~el[:N])&reflect] = np.abs(ion.sample(ni_reflect))
        vy[:N][reflect] = 0.0

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
        vxe = electron.source(n_pairs)
        vxi = ion.source(n_pairs)
        xe_new = np.random.rand(n_pairs)*L_source
        xi_new = np.random.rand(n_pairs)*L_source

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

        if save_xp:  xpa[i]  = xp
        if save_vx:  vxa[i]  = vx
        if save_vy:  vya[i]  = vy
        if save_E:   Ea[i]   = E0
        if save_phi: phia[i] = phi
        if save_rho: rhoa[i] = rho
        if save_sig: siga[i] = sigma
        if save_N:   Na[i]   = N
        if save_el:  ela[i]  = el

    results = {}
    if save_xp:  results['xp_all']  = xpa
    if save_vx:  results['vx_all']  = vxa
    if save_vy:  results['vy_all']  = vya
    if save_E:   results['E_all']   = Ea
    if save_phi: results['phi_all'] = phia
    if save_rho: results['rho_all'] = rhoa
    if save_sig: results['sig_all'] = siga
    if save_N:   results['N_all']   = Na
    if save_el:  results['el_all']  = ela

    results['N']   = N    
    results['xp']  = xp[:N]
    results['vx']  = vx[:N]
    results['vy']  = vy[:N]
    results['el']  = el[:N]
    results['E']   = E0
    results['phi'] = phi
    results['rho'] = rho
    results['sig'] = sigma


    return results
