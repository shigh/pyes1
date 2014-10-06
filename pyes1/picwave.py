import numpy as np
from pic1d2v import *

class PicWave(object):
    
    def __init__(self, N_e=2**8, nt=150, dt=0.2, nx=128,
                 L=2*np.pi, mode=1, A=.01, wp2=1., B0=0,
                 weight_method="CIC",
                 interp_method="CIC"):
        self.N_e  = N_e 
        self.nt   = nt
        self.dt   = dt
        self.nx   = nx
        self.L    = L
        self.mode = mode
        self.A    = A
        self.wp2  = wp2
        self.B0   = B0
        self.weight_method = weight_method
        self.interp_method = interp_method
        
    def init_run(self):
        """Called in run to set all of the computer values
        """
        self.dx       = self.L/self.nx
        self.N_i      = self.N_e
        self.x0       = np.linspace(0., self.L, self.N_e+1)[:-1]
        self.n0       = self.N_e/self.L
        self.k        = self.mode*2*np.pi/self.L
        self.x1       = self.A*np.cos(self.x0*self.k)
        self.init_pos = self.x0 + self.x1
        self.x0i      = np.linspace(0., self.L, self.N_i+1)[:-1]
        normalize(self.init_pos, self.L)

        self.q = self.wp2/self.n0
        self.m = self.q

        self.electron = Species(-self.q, self.m, self.N_e,
                                self.init_pos,
                                np.zeros(self.N_e), 
                                np.zeros(self.N_e))
        self.ion      = Species(+self.q, -1., self.N_i,
                                self.x0i,
                                np.zeros(self.N_i), 
                                np.zeros(self.N_i))

        self.species = [self.electron, self.ion]
        self.wp = np.sqrt(self.wp2)
        
    def run(self):

        self.init_run()
        self.xp, self.vx, self.vy, self.E, self.phi, self.rho = \
            pic(self.species, self.nx, self.dx, 
                self.nt, self.dt, self.L, self.B0, 
                solver_method="FFT",
                weight_method=self.weight_method,
                interp_method=self.interp_method)

    def delta_x(self):
        """Handle wrap around logic for delta x
        """
        L = self.L
        delta_x = self.xp[:,:self.N_e]-self.x0
        delta_x[delta_x>L/2.]  = delta_x[delta_x>L/2.]  - L
        delta_x[delta_x<-L/2.] = delta_x[delta_x<-L/2.] + L

        return delta_x

    def omega(self):
        pass

        
