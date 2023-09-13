### Generic imports
import os
import time
import math
import gym
import gym.spaces        as gsp
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

from   matplotlib.patches import Rectangle

###############################################
### Generic class
class sloshing(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, init=True,
                 L=2.5, amp=5.0, alpha=0.0005, g=9.81):

        # Main parameters
        self.L          = L                # length of domain
        self.nx         = 100*int(L)       # nb of discretization points
        self.dt         = 0.001            # timestep
        self.dt_act     = 0.05             # action timestep
        self.t_warmup   = 2.0              # warmup time
        self.t_act      = 10.0             # action time after warmup
        self.g          = g                # gravity
        self.n_obs      = int(self.nx/2)   # nb of obs pts
        self.amp        = amp              # amplitude scaling
        self.alpha      = alpha            # control penalization
        self.u_interp   = 0.01             # time on which action is interpolated
        self.blowup_rwd =-1.0              # reward in case of blow-up
        self.init_file  = "init_field.dat" # initialization file

        # Deduced parameters
        self.t_max      = self.t_warmup + self.t_act     # total simulation time
        self.dx         = float(self.L/self.nx)          # spatial step
        self.ndt_max    = int(self.t_max/self.dt)        # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)       # nb of numerical timesteps per action
        self.ndt_warmup = int(self.t_warmup/self.dt)     # nb of numerical timesteps for warmup
        self.n_act      = int(self.t_act/self.dt_act)    # nb of action steps per episode
        self.n_warmup   = int(self.t_warmup/self.dt_act) # nb of action steps for warmup
        self.n_interp   = int(self.u_interp/self.dt)     # nb of interpolation steps for action

        ### Path
        self.path        = "png"
        self.height_path = self.path+"/height"
        self.field_path  = self.path+"/field"
        self.action_path = self.path+"/action"
        os.makedirs(self.path,        exist_ok=True)
        os.makedirs(self.height_path, exist_ok=True)
        os.makedirs(self.field_path,  exist_ok=True)
        os.makedirs(self.action_path, exist_ok=True)

        ### Declare arrays
        self.x       = np.linspace(0, self.nx, num=self.nx, endpoint=False)*self.dx
        self.h       = np.zeros((self.nx+2)) # current h
        self.q       = np.zeros((self.nx+2)) # current q
        self.v       = np.zeros((self.nx+2)) # current v
        self.qgh     = np.zeros((self.nx+2)) # current q**2/h + 0.5*g*h**2
        self.fhg     = np.zeros((self.nx+2)) # left  flux for h
        self.fhd     = np.zeros((self.nx+2)) # right flux for h
        self.fqg     = np.zeros((self.nx+2)) # left  flux for q
        self.fqd     = np.zeros((self.nx+2)) # right flux for q
        self.rhsh    = np.zeros((self.nx+2)) # rhs for h
        self.rhsq    = np.zeros((self.nx+2)) # rhs for q
        self.rhshp   = np.zeros((self.nx+2)) # previous rhs for h
        self.rhsqp   = np.zeros((self.nx+2)) # previous rhs for q
        self.c       = np.zeros((self.nx+1)) # rusanov parameter

        self.h_init  = np.zeros((self.nx+2))   # h initialization
        self.q_init  = np.zeros((self.nx+2))   # q initialization

        # Load initialization file
        if (init): self.load(self.init_file)

        # Define action space
        self.action_space = gsp.Box(low   =-1.0,
                                    high  = 1.0,
                                    shape = (1,),
                                    dtype = np.float32)

        # Define observation space
        self.h_min =-1.0
        self.h_max = 1.0

        low  = np.ones((self.n_obs))*self.h_min
        high = np.ones((self.n_obs))*self.h_max

        self.observation_space = gsp.Box(low   =-low,
                                         high  = high,
                                         shape = (self.n_obs,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()
        self.h[:] = self.h_init[:]
        self.q[:] = self.q_init[:]

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.h[:]   = 1.0
        self.q[:]   = 0.0
        self.v[:]   = 0.0

        # Other fields
        self.qgh[:]   = 0.0
        self.fhg[:]   = 0.0
        self.fhd[:]   = 0.0
        self.fqg[:]   = 0.0
        self.fqd[:]   = 0.0
        self.rhsh[:]  = 0.0
        self.rhsq[:]  = 0.0
        self.rhshp[:] = 0.0
        self.rhsqp[:] = 0.0

        # Actions
        self.u  = [0.0]
        self.up = [0.0]

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Run warmup
    def warmup(self):

        # Run until flow is developed
        for i in range(self.n_warmup):
            self.solve()

    # Define excitation signal
    def signal(self, t, dt):

        a = 0.5*(np.cos(np.pi*t) + 3.0*np.cos(4.0*np.pi*t))

        return a

    # Step
    def step(self, u=None):

        # Run solver
        self.solve(u)

        # Check end of episode
        done  = False
        trunc = False
        if (self.stp == self.n_act-1):
            done  = True
            trunc = True
        if (np.any((self.h < -5.0*self.h_max) | (self.h > 2.0*self.h_max))):
            print("Blowup")
            done  = True
            trunc = False
            rwd   = self.blowup_rwd

        # Retrieve data
        obs = self.get_obs()
        rwd = self.get_rwd(done)

        # Update step
        self.stp += 1

        return obs, rwd, done, trunc, None

    # Resolution
    def solve(self, u=None):

        if (u is None): u = self.u.copy()

        # Save actions
        self.up[:] = self.u[:]
        self.u[:]  = u[:]

        nx = self.nx
        dx = self.dx

        # Run solver
        for i in range(self.ndt_act):

            # Boundary conditions
            self.h[0] = self.h[1]
            self.q[0] = 0.0
            self.h[self.nx+1] = self.h[self.nx]
            self.q[self.nx+1] = 0.0

            # Update previous fields
            self.rhshp[1:nx+1] = self.rhsh[1:nx+1]
            self.rhsqp[1:nx+1] = self.rhsq[1:nx+1]

            # Compute v and q**2/h + 0.5*h**2 terms
            self.v[:]   = self.q[:]/self.h[:]
            self.qgh[:] = self.q[:]**2/self.h[:] + 0.5*self.g*self.h[:]**2

            # Compute rusanov parameter
            self.c[0:nx+1] = np.maximum(
                np.abs(self.v[0:nx+1]) + np.sqrt(self.g*self.h[0:nx+1]),
                np.abs(self.v[1:nx+2]) + np.sqrt(self.g*self.h[1:nx+2]))

            # Compute fluxes for h
            rusanov(self.fhg[1:nx+1], self.q[0:nx],   self.q[1:nx+1],
                    self.h[0:nx],     self.h[1:nx+1], self.c[0:nx])
            rusanov(self.fhd[1:nx+1], self.q[1:nx+1], self.q[2:nx+2],
                    self.h[1:nx+1],   self.h[2:nx+2], self.c[1:nx+1])

            # Compute fluxes for q
            rusanov(self.fqg[1:nx+1], self.qgh[0:nx],   self.qgh[1:nx+1],
                    self.q[0:nx],     self.q[1:nx+1],   self.c[0:nx])
            rusanov(self.fqd[1:nx+1], self.qgh[1:nx+1], self.qgh[2:nx+2],
                    self.q[1:nx+1],   self.q[2:nx+2],   self.c[1:nx+1])

            # Form right-hand-sides
            self.rhsh[1:nx+1] = (self.fhd[1:nx+1] - self.fhg[1:nx+1])/dx
            self.rhsq[1:nx+1] = (self.fqd[1:nx+1] - self.fqg[1:nx+1])/dx

            # Add control
            alpha = min(float(i)/float(self.n_interp), 1.0)
            u     = (1.0-alpha)*self.up[0] + alpha*self.u[0]
            self.rhsq[1:nx+1] += u*self.amp

            # March in time
            adams(self.h, self.rhsh, self.rhshp, self.nx, self.dt)
            adams(self.q, self.rhsq, self.rhsqp, self.nx, self.dt)

    # Retrieve observations
    def get_obs(self):

        obs = self.q.copy()[1:-1]
        obs = obs[::2]

        return obs

    # Compute reward
    def get_rwd(self, done):

        rwd      = 0.0
        hdiff    = np.zeros((self.nx))
        hdiff[:] = self.h[1:self.nx+1] - 1.0
        hrwd     = np.linalg.norm(hdiff)*self.dx
        rwd     -= hrwd
        rwd     -= self.alpha*abs(self.amp*self.u[0])

        return rwd

    # Render environment
    def render(self, mode="human", show=False, dump=True):

        # Plot field
        plt.clf()
        plt.cla()
        fig = plt.figure(figsize=(7,3))
        ax  = fig.add_subplot(20, 1, (1,17))
        ax.set_xlim([0.0,self.L])
        ax.set_ylim([0.25,1.75])
        ax.set_xticks([])
        ax.set_yticks([])

        plt.plot(np.ones_like(self.x), color='k', lw=1, linestyle='dashed')
        plt.plot(self.x, self.h[1:self.nx+1])

        # Plot control
        ax = fig.add_subplot(20, 1, (19,20))
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([ 0.0, 0.2])
        ax.set_xticks([])
        ax.set_yticks([])
        x = 0.0
        y = 0.05
        color = 'r' if self.u[0] > 0.0 else 'b'
        ax.add_patch(Rectangle((x,y), 0.98*self.u[0], 0.1,
                               color=color, fill=True, lw=2))

        # Save figure
        filename = self.height_path+'/'+str(self.stp_plot)+'.png'
        fig.savefig(filename, dpi=100, bbox_inches="tight")
        if show: plt.pause(0.01)
        plt.close()

        # Dump
        if dump: self.dump(self.field_path+"/field_"+str(self.stp_plot)+".dat",
                           self.action_path+"/jet_"+str(self.stp_plot)+".dat")

        self.stp_plot += 1

    # Dump (h,q)
    def dump(self, field_name, control_name=None):

        array = self.x.copy()
        array = np.vstack((array, self.h[1:self.nx+1]))
        array = np.vstack((array, self.q[1:self.nx+1]))
        array = np.transpose(array)

        np.savetxt(field_name, array,  fmt='%.5e')
        if (control_name is not None):
            np.savetxt(control_name, self.u, fmt='%.5e')

    # Load (h,q)
    def load(self, filename):

        f = np.loadtxt(filename)
        self.h_init[1:self.nx+1] = f[:,1]
        self.q_init[1:self.nx+1] = f[:,2]

    # Closing
    def close(self):
        pass

###############################################
# rusanov flux
@nb.njit(cache=True)
def rusanov(f, fug, fud, ug, ud, c):

    f[:] = 0.5*(fug[:] + fud[:]) - 0.5*c[:]*(ud[:] - ug[:])

# 2nd order adams-bashforth update in time
@nb.njit(cache=True)
def adams(u, rhs, rhsp, nx, dt):

    u[1:nx+1] += 0.5*dt*(-3.0*rhs[1:nx+1] + rhsp[1:nx+1])

