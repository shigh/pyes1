import numpy as np
import unittest
from pic1d2v import *

# Use the max norm
norm = lambda x: np.max(np.abs(x))

class TestCalcE(unittest.TestCase):
    k1 = 2*np.pi
    
    def get_x(self, N):
        return np.linspace(0, 1, N+1)[:-1]
        
    def get_u(self, x_vals):
        return np.sin(self.k1*x_vals)
        
    def get_exact(self, x_vals):
        return -self.k1*np.cos(self.k1*x_vals)
        
    def get_error(self, N):
        x_vals = self.get_x(N)
        dx     = x_vals[1] - x_vals[0]
        u      = self.get_u(x_vals)
        exact  = self.get_exact(x_vals)
        E      = calc_E(u, dx)
        return norm(E - exact)
        
    def test_converges(self):
        N = 10000
        tol = 10e-5
        error = self.get_error(N)
        self.assertLess(error, tol)

    def test_order(self):
        tol = .001
        k_vals = [10, 11]
        errors = [self.get_error(2**k) for k in k_vals]
        order = np.log2(errors[0]) - np.log2(errors[1])
        self.assertLess(np.abs(order-2), tol)

        
class TestLeapFrog(unittest.TestCase):

    def leap_frog(self, xp0, x0, w0, nt, dt, L, wc):
        
        nx = len(xp0)
        w02 = w0**2
        a   = -w02*np.ones(nx)
        xpa = np.zeros((nt+1, nx))
        xp  = xp0.copy()
        vx  = np.zeros_like(xp)
        vy  = np.zeros_like(xp)

        xpa[0] = xp0
        rotate(vx, vy, -wc, dt)
        accel(vx, vy, xp-x0, -a, dt)

        for i in range(1, nt+1):
            # Update velocity
            accel(vx, vy, xp-x0, a, dt)
            rotate(vx, vy, wc, dt)
            accel(vx, vy, xp-x0, a, dt)
            # Update position
            move(xp, vx, vy, dt, L)
            xpa[i] = xp

        return xpa

    def test_two_steps(self):
        """Two accel calls should give us one full time step
        Using B=0
        """
        vx = np.linspace(0, 1, 10)
        vy = np.zeros_like(vx)
        E = vx.copy()
        alpha, dt = .5, .1
        expected = vx + alpha*E*dt
        
        accel(vx, vy, E, alpha, dt)
        rotate(vx, vy, 0., dt)
        accel(vx, vy, E, alpha, dt)
        
        error = np.linalg.norm(vx-expected)
        self.assertAlmostEqual(error, 0.)
        
    def test_no_rotation(self):
        """When B=wc=0 rotate should do nothing
        """
        vx0 = np.linspace(0, 1, 10)
        vy0 = np.linspace(1, 2, 10)
        vx, vy = vx0.copy(), vy0.copy()

        rotate(vx, vy, 0., .1)

        self.assertTrue((vx==vx0).all())
        self.assertTrue((vy==vy0).all())

    def test_half_step_back_reversible(self):
        """The initial half step back in v should be reversible
        """
        vx0 = np.linspace(0, 1, 10)
        vy0 = np.linspace(1, 2, 10)
        vx, vy, E = vx0.copy(), vy0.copy(), vx0.copy()
        wc, dt, alpha = .1, .1, .5
        
        rotate(vx, vy, -wc, dt)
        accel(vx, vy, E, -alpha, dt)
        accel(vx, vy, E,  alpha, dt)
        rotate(vx, vy, wc, dt)
        
        self.assertAlmostEqual(np.linalg.norm(vx-vx0), 0.)
        self.assertAlmostEqual(np.linalg.norm(vy-vy0), 0.)
        
    def test_normalize(self):
        """Move particles out of [0, L) back into [0, L)
        """
        L = 1.
        # The first value (-10e-20) should get washed out
        # and rounded to 0, not to L.
        x  = np.array([-10e-20, -1, -.5, -1e-10, 0, .5, L-1e-10, L, L+.5])
        y  = np.array([0, 0, .5, L-1e-10, 0, .5, L-1e-10, 0, .5])
        normalize(x, L)
        
        self.assertTrue((x==y).all())
        self.assertTrue((x>=0).all())
        self.assertTrue((x<L).all())
        
    def test_move(self):
        """Everything gets pushed when do_move=None
        """
        L   = 1
        dt  = .5
        xp  = np.linspace(0, L, 20+1)[:-1]
        vx  = np.ones_like(xp)*.2
        vy  = None
        xpe = xp+dt*L*.2
        normalize(xpe, L)
        move(xp, vx, vy, dt, L)

        self.assertTrue((xp==xpe).all())

    def test_move_do_move(self):
        """Correctly moves only a subset of xp when do_move!=None
        """
        L   = 1
        dt  = .5
        do_move = np.ndarray(20, dtype=np.bool)
        do_move[:10] = True
        do_move[10:] = False
        xp  = np.linspace(0, L, 20+1)[:-1]
        vx  = np.ones_like(xp)*.2
        vy  = None
        xpe = xp+dt*L*.2
        normalize(xpe, L)
        xpe[10:] = xp[10:]
        move(xp, vx, vy, dt, L, do_move=do_move)

        self.assertTrue((xp==xpe).all())

    def test_leap_frog_order(self):
        nx  = 10
        L   = 1.
        x0 = L/2.
        xp0 = np.linspace(0, L/4., nx+1)[1:]+x0
        w0  = 1.
        wc  = 0
        T   = 10.
        tol = 0.001
        k_vals = range(9, 11)
        errors = []
        for k in k_vals:
            nt = 2**k
            dt = T/nt
            t_vals = np.linspace(0, T, nt+1)
            expected = np.zeros((nt+1, nx))
            for j in range(nx):
                expected[:, j] = (xp0-x0)[j]*np.cos(w0*t_vals)+x0

            xpa = self.leap_frog(xp0, x0, w0, nt, dt, L, wc)

            errors.append([norm(expected[:,j]-xpa[:,j])
                           for j in range(nx)])

        e2 = np.log2(np.array(errors))
        order = e2[:-1]-e2[1:]
        self.assertTrue((np.abs(order-2)<tol).all())

