### Generic imports
import os
import time
import math
import random
import gym
import gym.spaces        as gsp
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.cm     as cm
import numba             as nb

from   matplotlib.patches import Rectangle

###############################################
### Burgers environment
class burgers(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, init=True,
                 n_sgts=10, nu=0.002):

        # Main parameters
        self.L          = 1.0    # domain length
        self.nx         = 100    # nb of discretization pts
        self.dx         = 0.01
        self.t_max      = 6.0    # total simulation time
        self.dt         = 0.005
        self.dt_act     = 0.05   # action timestep
        self.n_sgts     = n_sgts # nb of action segments
        self.n_obs_pts  = 20     # nb of observation pts
        self.nu         = nu     # viscosity
        self.scale      = 1.0    # action scaling
        self.x0         = 0.5    # center of initial bump
        self.sigma      = 0.05    # variance of initial bump
        self.c          = 0.1    # matching velocity

        # Deduced parameters
        self.nop        = 5
        self.n_obs_tot  = self.n_obs_pts + 2*self.nop          # total nb of obs
        #self.dx         = float(self.L/self.nx)       # spatial step
        #self.dt         = 0.5*self.dx                 # timestep
        self.ndt_max    = int(self.t_max/self.dt)     # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)    # nb of numerical timesteps per action
        self.t_act      = self.t_max                  # action time
        self.n_act      = int(self.t_act/self.dt_act) # nb of action steps per episode
        self.nx_sgts    = self.nx//self.n_sgts        # nb of pts in each segment
        self.nx_obs     = self.nx//self.n_obs_pts     # nb of pts between each observation pt in x direction

        # Arrays
        self.x   = np.linspace(0, self.nx, num=self.nx, endpoint=False)*self.dx
        self.u   = np.zeros((self.nx))
        self.up  = np.zeros((self.nx))
        self.upp = np.zeros((self.nx))
        self.du  = np.zeros((self.nx))
        self.ddu = np.zeros((self.nx))
        self.rhs = np.zeros((self.nx))
        self.uex = np.zeros((self.nx))

        # Define action space
        self.action_space = gsp.Box(low   =-self.scale,
                                    high  = self.scale,
                                    shape = (self.n_sgts,),
                                    dtype = np.float32)

        # Define observation space
        low =  np.zeros((self.n_obs_tot))
        high = np.ones((self.n_obs_tot))
        self.observation_space = gsp.Box(low   = low,
                                         high  = high,
                                         shape = (self.n_obs_tot,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.t = 0.0
        window(self.u,   self.x, 0.0, self.c, self.x0, self.sigma, self.L)
        window(self.up,  self.x, 0.0, self.c, self.x0, self.sigma, self.L)
        window(self.upp, self.x, 0.0, self.c, self.x0, self.sigma, self.L)

        # Other fields
        self.du[:]  = 0.0
        self.ddu[:] = 0.0
        self.rhs[:] = 0.0
        self.uex[:] = 0.0

        # Actions
        self.a  = [0.0]*self.n_sgts
        self.ap = [0.0]*self.n_sgts

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Step
    def step(self, a=None):

        # Run solver
        self.solve(a)

        # Retrieve data
        obs = self.get_obs()
        rwd = self.get_rwd()

        # Check end of episode
        done  = False
        trunc = False
        if (self.stp == self.n_act-1):
            done  = True
            trunc = True

        # Update step
        self.stp += 1

        return obs, rwd, done, trunc, None

    # Resolution
    def solve(self, a=None):

        if (a is None): a = self.a.copy()

        # Save actions
        self.ap = self.a
        self.a  = a

        # Run solver
        for i in range(self.ndt_act):

            # Update previous field
            self.upp[:] = self.up[:]
            self.up[:]  = self.u[:]

            # Boundary conditions
            self.u[-1] = self.u[1]
            self.u[0]  = self.u[-2]

            #self.u[-1] = 0.0
            #self.u[0]  = 0.0

            # Compute spatial derivative
            derx(self.u,  self.du,  self.nx, self.dx)
            derxx(self.u, self.ddu, self.nx, self.dx)

            # Build rhs
            rhs(self.u, self.du, self.ddu, self.nu, self.rhs)

            # Add control
            for k in range(self.n_sgts):
                s = k*self.nx_sgts
                e = (k+1)*self.nx_sgts
                self.rhs[s:e] += self.a[k]

            # Update
            dert(self.u, self.up, self.upp, self.rhs, self.nx, self.dt)
            self.t += self.dt

    # Retrieve observations
    def get_obs(self):

        # Fill new observations
        obs = np.zeros((1,self.n_obs_tot))
        j   = self.nx_obs//2
        for i in range(self.n_obs_pts):
            obs[0,i+2] = self.u[j+i*self.nx_obs]

        obs[0, :self.nop] = self.u[-self.nop:]
        obs[0,-self.nop:] = self.u[ :self.nop]
            #obs[1,i] = self.rhs[j+i*self.nx_obs]

        obs = np.reshape(obs, [-1])

        return obs

    # Compute reward
    def get_rwd(self):

        window(self.uex, self.x, self.t, self.c, self.x0, self.sigma, self.L)
        rwd =-np.sum(np.absolute(self.u[:] - self.uex[:]))*self.dx

        #rwd = 1.0e-5*np.sum(self.ddu[:]**2)*self.dx

        return rwd

    # Rendering
    def render(self, mode="human", show=False, dump=True):

        # Open directories
        if (self.stp_plot == 0):
            self.path = "png"
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path+"/gif", exist_ok=True)

        # Plot field
        fig = plt.figure(figsize=(7,3))
        ax  = fig.add_subplot(20, 1, (1,17))
        ax.set_xlim([0.0,self.L])
        ax.set_ylim([-0.1,1.5])
        ax.set_xticks([])
        ax.set_yticks([])
        window(self.uex, self.x, self.t, self.c, self.x0, self.sigma, self.L)
        plt.plot(self.x, self.u)
        plt.plot(self.x, self.uex)

        # Plot control
        ax = fig.add_subplot(20, 1, (19,20))
        ax.set_xlim([ 0.0, self.L])
        ax.set_ylim([-self.scale, self.scale])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.plot(np.zeros_like(self.x), color='k', lw=1, linestyle='dashed')

        for i in range(self.n_sgts):
            x = (0.5*self.nx_sgts + i*self.nx_sgts)*self.dx
            y = 0.0
            s = (0.2*self.nx_sgts)*self.dx
            color = 'r' if self.a[i] > 0.0 else 'b'
            ax.add_patch(Rectangle((x,y), s, 0.98*self.a[i],
                                   color=color, fill=True, lw=1))

        # Save figure
        filename = self.path+"/gif/"+str(self.stp_plot)+".png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        if show: plt.pause(0.01)
        plt.close()

        # Dump
        #if dump: self.dump(self.path+"/lorenz.dat")

        self.stp_plot += 1

    # Dump x, y, z
    # def dump(self, filename):

    #     array = self.ht.copy()
    #     array = np.vstack((array, self.hx[:,0]))
    #     array = np.vstack((array, self.hx[:,1]))
    #     array = np.vstack((array, self.hx[:,2]))
    #     array = np.transpose(array)

    #     np.savetxt(filename, array, fmt='%.5e')

    # Close environment
    def close(self):
        pass

###############################################
# Window signal
def window(u, x, t, c, x0, sg, L):

    for i in range(len(x)):
        xx = x[i] - x0 - c*t
        while (xx + x0 < 0.0): xx += L
        u[i] = 0.5*np.exp(-(xx/(2.0*sg))**2)

# 1st derivative tvd scheme
@nb.njit(cache=True)
def derx(u, du, nx, dx):

    fp        = np.zeros((nx))
    fm        = np.zeros((nx))
    phi       = np.zeros((nx))
    r         = np.zeros((nx))

    r[1:nx-1]   = (u[1:nx-1] - u[0:nx-2])/(u[2:nx] - u[1:nx-1] + 1.0e-8)
    #phi[1:nx-1] = np.maximum(0.0, np.minimum(r[1:nx-1], 1.0))
    #phi[1:nx-1] = (r[1:nx-1] + np.absolute(r[1:nx-1]))/(1.0 + r[1:nx-1])
    phi[1:nx-1] = np.maximum(np.maximum(0.0, np.minimum(2.0*r[1:nx-1], 1.0)), np.minimum(r[1:nx-1],2.0))

    fp[1:nx-1] = u[1:nx-1] + 0.5*phi[1:nx-1]*(u[2:nx]   - u[1:nx-1]) # f_m+1/2
    fm[1:nx-1] = u[0:nx-2] + 0.5*phi[0:nx-2]*(u[1:nx-1] - u[0:nx-2]) # f_m-1/2
    du[1:nx-1] = (fp[1:nx-1] - fm[1:nx-1])/dx

# 2nd derivative, 2nd order
@nb.njit(cache=True)
def derxx(u, ddu, nx, dx):

    ddu[1:nx-1] = (u[2:nx] + u[0:nx-2] - 2.0*u[1:nx-1])/(dx*dx)

# time derivative
@nb.njit(cache=True)
def dert(u, up, upp, rhs, nx, dt):

    u[1:nx-1] = (4.0*up[1:nx-1] - upp[1:nx-1] - 2.0*dt*rhs[1:nx-1])/3.0

# time derivative
@nb.njit(cache=True)
def rhs(u, du, ddu, nu, r):

    r[1:-1] = u[1:-1]*du[1:-1] - nu*ddu[1:-1]
    #r[1:-1] = 1.0*du[1:-1] #+ nu*ddu[1:-1]
