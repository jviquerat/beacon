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

from   matplotlib.patches import Rectangle, Circle

###############################################
### Burgers environment
class burgers(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, init=True,
                 u_target=0.5, scale=10.0, sigma=0.1, ctrl_pos=0.3):

        # Main parameters
        self.L          = 1.0      # domain length
        self.nx         = 500      # nb of discretization pts
        self.t_max      = 10.0     # total simulation time
        self.dt_act     = 0.05     # action timestep
        self.n_obs_pts  = 20       # nb of observation pts
        self.scale      = scale    # action scaling
        self.sigma      = sigma    # variance of noise
        self.u_target   = u_target # target value
        self.ctrl_xpos  = ctrl_pos # position of control point

        # Deduced parameters
        self.dx         = float(self.L/self.nx)         # spatial step
        self.ctrl_pos   = int(self.ctrl_xpos/self.dx)   # position of control point
        self.dt         = 0.2*self.dx                  # timestep
        self.ndt_max    = int(self.t_max/self.dt)       # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)      # nb of numerical timesteps per action
        self.t_act      = self.t_max                    # action time
        self.n_act      = int(self.t_act/self.dt_act)   # nb of action steps per episode
        self.nx_obs     = self.ctrl_pos//self.n_obs_pts # nb of pts between each observation pt

        # Arrays
        self.x   = np.linspace(0, self.nx, num=self.nx, endpoint=False)*self.dx
        self.u   = np.zeros((self.nx))
        self.up  = np.zeros((self.nx))
        self.upp = np.zeros((self.nx))
        self.du  = np.zeros((self.nx))
        self.rhs = np.zeros((self.nx))
        self.uex = np.zeros((self.nx))

        # Define action space
        self.action_space = gsp.Box(low   =-self.scale,
                                    high  = self.scale,
                                    shape = (1,),
                                    dtype = np.float32)

        # Define observation space
        low =  np.zeros((self.n_obs_pts))
        high = np.ones((self.n_obs_pts))
        self.observation_space = gsp.Box(low   = low,
                                         high  = high,
                                         shape = (self.n_obs_pts,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.t      = 0.0
        self.u[:]   = self.u_target
        self.up[:]  = self.u_target
        self.upp[:] = self.u_target

        # Other fields
        self.du[:]  = 0.0
        self.rhs[:] = 0.0
        self.uex[:] = 0.0

        # Actions
        self.a  = [0.0]

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
        self.a[:] = a[:]

        noise = np.random.uniform(-self.sigma, self.sigma, 1)

        # Run solver
        for i in range(self.ndt_act):

            # Update previous field
            self.upp[:] = self.up[:]
            self.up[:]  = self.u[:]

            # Boundary conditions
            self.u[0]         = self.u_target + noise
            self.u[self.nx-1] = self.u[self.nx-2]

            # Compute spatial derivative
            derx(self.u,  self.du,  self.nx, self.dx)

            # Build rhs
            rhs(self.u, self.du, self.rhs, self.nx)

            # Add control
            self.rhs[self.ctrl_pos] += self.a[0]

            # Update
            dert(self.u, self.up, self.upp, self.rhs, self.nx, self.dt)
            self.t += self.dt

    # Retrieve observations
    def get_obs(self):

        # Fill new observations
        obs    = np.zeros((self.n_obs_pts))
        obs[:] = self.u[self.ctrl_pos-self.n_obs_pts:self.ctrl_pos]

        return obs

    # Compute reward
    def get_rwd(self):

        rwd = -np.sum(np.abs(self.u[self.ctrl_pos:] - self.u_target))*self.dx

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
        ax.set_ylim([0.3,0.7])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axvline(x=self.ctrl_xpos, color='k', lw=1)
        plt.plot(np.ones_like(self.x)*self.u_target,
                 color='k', lw=1, linestyle='dashed')
        plt.plot(self.x, self.u)

        # Plot control
        ax = fig.add_subplot(20, 1, (19,20))
        ax.set_xlim([-self.scale, self.scale])
        ax.set_ylim([ 0.0, 0.2])
        ax.set_xticks([])
        ax.set_yticks([])
        x = 0.0
        y = 0.05
        color = 'r' if self.a[0] > 0.0 else 'b'
        ax.add_patch(Rectangle((x,y), 0.98*self.a[0], 0.1,
                               color=color, fill=True, lw=2))

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
# 1st derivative tvd scheme
@nb.njit(cache=True)
def derx(u, du, nx, dx):

    #du[1:nx-1] = (u[1:nx-1] - u[0:nx-2])/dx

    fp        = np.zeros((nx))
    fm        = np.zeros((nx))
    phi       = np.zeros((nx))
    r         = np.zeros((nx))

    #r[1:nx-1]   = (u[1:nx-1] - u[0:nx-2])/(u[2:nx] - u[1:nx-1] + 1.0e-8)
    phi[1:nx-1] = np.maximum(0.0, np.minimum(r[1:nx-1], 1.0))
    #phi[1:nx-1] = (r[1:nx-1] + np.absolute(r[1:nx-1]))/(1.0 + r[1:nx-1])
    #phi[1:nx-1] = np.maximum(np.maximum(0.0, np.minimum(2.0*r[1:nx-1], 1.0)), np.minimum(r[1:nx-1],2.0))

    fp[1:nx-1] = u[1:nx-1] + 0.5*phi[1:nx-1]*(u[2:nx]   - u[1:nx-1]) # f_m+1/2
    fm[1:nx-1] = u[0:nx-2] + 0.5*phi[0:nx-2]*(u[1:nx-1] - u[0:nx-2]) # f_m-1/2
    du[1:nx-1] = (fp[1:nx-1] - fm[1:nx-1])/dx

# time derivative
@nb.njit(cache=True)
def dert(u, up, upp, rhs, nx, dt):

    u[1:nx-1] = (4.0*up[1:nx-1] - upp[1:nx-1] - 2.0*dt*rhs[1:nx-1])/3.0

# time derivative
@nb.njit(cache=True)
def rhs(u, du, r, nx):

    r[1:nx-1] = u[1:nx-1]*du[1:nx-1]

    #r[1:nx-1] = (1.0-u[1:nx-1])*du[1:nx-1]
