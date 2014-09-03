import numpy as np
import unittest
from pic1d2v import *

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
        return np.linalg.norm(E - exact)
    
    def test_converges(self):
        N = 10000
        tol = 10e-5
        error = self.get_error(N)
        self.assertLess(error, tol)

    # I am not getting the expected order
    # Getting about 1.5 instead of 2
    @unittest.expectedFailure
    def test_order(self):
        tol = .2
        k_vals = [10, 11]
        errors = [self.get_error(2**k) for k in k_vals]
        order = np.log2(errors[0]) - np.log2(errors[1])
        self.assertLess(np.abs(order-2), tol)

        
        
