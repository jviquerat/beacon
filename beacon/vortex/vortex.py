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
from   scipy.interpolate  import interp1d

###############################################
### Vortex environment
class vortex(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0,
                 re=50.0):

        # Main parameters

        self.lmbda_re   = 9.153
        self.lmbda_cx   = 3.239
        self.mu_re      = 308.9
        self.mu_cx      =-1025.0
        self.alpha_re   = 0.03492
        self.alpha_cx   = 0.01472
        self.beta       = 1.0
        self.re         = re
        self.re_crit    = 46.6
        self.ire        = 1.0/self.re_crit - 1.0/self.re
        self.omega_s    = 1.1
        self.omega_f    = 0.74
        self.domega     = self.omega_s - self.omega_f

        self.gamma      = 0.023
        self.mass       = 10.0
        self.weight     = 2.0
        self.beta_m     = self.beta/(self.omega_f*self.mass)

        self.dt         = 0.5   # timestep
        self.dt_act     = 25.0   # action timestep
        self.t_max      = 5000.0 # total simulation time

        # Deduced parameters
        self.n_obs      = 8                           # total nb of observations
        self.ndt_max    = int(self.t_max/self.dt)     # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)    # nb of numerical timesteps per action
        self.t_act      = self.t_max                  # action time
        self.n_act      = int(self.t_act/self.dt_act) # nb of action steps per episode

        # Arrays
        self.x  = np.zeros(4) # unknowns (ar, ai, yr, yi)
        self.xk = np.zeros(4) # lsrk storage
        self.fx = np.zeros(4) # rhs

        # Initialize integrator
        self.integrator = lsrk4()

        # Define action space
        self.action_space = gsp.Box(low   =-1.0,
                                    high  = 1.0,
                                    shape = (2,),
                                    dtype = np.float32)

        # Define observation space
        low  = np.zeros((self.n_obs))
        high = np.ones( (self.n_obs))
        self.observation_space = gsp.Box(low   = low,
                                         high  = high,
                                         shape = (self.n_obs,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.t    = 0.0
        self.x[0] = 0.01
        self.x[1] = 0.01
        self.x[2] = 0.01
        self.x[3] = 0.01

        # Initial physical y value for reward computation
        self.y   = 2.0*(self.x[2]*math.cos(self.omega_f*self.t) -
                        self.x[3]*math.sin(self.omega_f*self.t))

        # Other fields
        self.xk[:] = 0.0
        self.fx[:] = 0.0
        self.hx    = np.empty((0, 4))
        self.ht    = np.empty((0))

        # Copy first step
        self.hx = np.append(self.hx, np.array([self.x]), axis=0)
        self.ht = np.append(self.ht, np.array([self.t]), axis=0)

        # Actions
        self.u = np.zeros(2)

        # Observations
        self.obs = np.zeros((2, 4))

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Step
    def step(self, u=None):

        # Run solver
        self.solve(u)

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
    def solve(self, u=None):

        if (u is None): u = self.u.copy()

        # Save actions
        self.u = u.copy()
        kmod   = self.u[0]
        kphase = self.u[1]

        #kmod   = 0.0408
        #kphase = -0.164

        # Run solver
        for i in range(self.ndt_act):

            # Update intermediate storage
            self.xk[:] = self.x[:]

            # Loop on integration steps
            for j in range(self.integrator.steps()):

                # Compute rhs
                self.fx[0] = self.ire*(self.lmbda_re*self.x[0] - self.lmbda_cx*self.xk[1]) - (self.mu_re*self.x[0] - self.mu_cx*self.x[1])*(self.x[0]**2 + self.x[1]**2) + (self.alpha_re*self.x[2] - self.alpha_cx*self.x[3])

                self.fx[1] = self.ire*(self.lmbda_re*self.x[1] + self.lmbda_cx*self.xk[0]) - (self.mu_re*self.x[1] + self.mu_cx*self.x[0])*(self.x[0]**2 + self.x[1]**2) + (self.alpha_re*self.x[3] + self.alpha_cx*self.x[2])

                self.fx[2] =-self.omega_f*self.gamma*self.x[2] - self.domega*self.x[3] + self.beta_m*self.x[0] + self.x[0]*kmod*math.cos(kphase) - self.x[1]*kmod*math.sin(kphase)

                self.fx[3] = -self.omega_f*self.gamma*self.x[3] + self.domega*self.x[2] + self.beta_m*self.x[1] + self.x[0]*kmod*math.sin(kphase) + self.x[1]*kmod*math.cos(kphase)

                # Update
                self.integrator.update(self.x, self.xk, self.fx, j, self.dt)

            # Update main storage
            self.x[:] = self.xk[:]

            # Store unknowns
            self.t += self.dt
            self.hx = np.append(self.hx, np.array([self.x]), axis=0)
            self.ht = np.append(self.ht, np.array([self.t]), axis=0)

    # Retrieve observations
    def get_obs(self):

        # Fill new observations
        self.obs[0,:] = self.x[:]
        self.obs[1,:] = self.fx[:]

        obs = np.reshape(self.obs, [-1])

        return obs

    # Compute reward
    def get_rwd(self):

        self.yp = self.y
        self.y  = 2.0*(self.x[2]*math.cos(self.omega_f*self.t) -
                       self.x[3]*math.sin(self.omega_f*self.t))
        rwd = 2.0*self.omega_s*self.gamma*((self.y - self.yp)/self.dt)**2

        return rwd

    # Rendering
    def render(self, mode="human", show=False, dump=True):

        # Open directories
        if (self.stp_plot == 0):
            self.path = "png"
            os.makedirs(self.path, exist_ok=True)
            os.makedirs(self.path+"/gif", exist_ok=True)

        # Full plot at the end of the episode
        if (self.stp == self.n_act):
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(8,2))
            fig.tight_layout()
            plt.plot(self.ht[:], self.hx[:,0])
            ax.set_xlim([0.0, self.t_max])
            ax.set_ylim([-20.0, 20.0])

            filename = self.path+"/history.png"
            plt.grid()
            plt.savefig(filename, dpi=100)
            plt.close()

        # Plot multiple pngs to generate gif
        plt.clf()
        plt.cla()
        fig = plt.figure(tight_layout=True)
        ax  = fig.add_subplot(15, 1, (1,14))#, projection='3d')
        ax.set_axis_off()
        fig.tight_layout()
        #ax.set_xlim([-1.0, 1.0])
        #ax.set_ylim([-1.0, 1.0])
        #ax.set_zlim([  0.0, 40.0])

        #if (len(self.ht) > 2):
        #    fx = interp1d(self.ht, self.hx[:,0], kind="quadratic")
        #    fy = interp1d(self.ht, self.hx[:,1], kind="quadratic")
            #fz = interp1d(self.ht, self.hx[:,2], kind="quadratic")

        #    n  = len(self.ht)
        #    tt = np.linspace(self.ht[0], self.ht[-1], 5*n)
        #    hhx = fx(tt)
        #    hhy = fy(tt)
            #hhz = fz(tt)

        #    ax.plot(hhx, hhy, linewidth=1)
        #else:
        ax.plot(self.hx[:,0],
                self.hx[:,1],
                linewidth=1)

        # Plot control
        ax = fig.add_subplot(15, 1, 15)
        fig.tight_layout()
        ax.set_xlim([-1.0, 1.0])
        ax.set_ylim([ 0.0, 0.2])
        ax.set_xticks([])
        ax.set_yticks([])
        x = 0.0
        y = 0.05
        #color = 'r' if self.u > 0.0 else 'b'
        #ax.add_patch(Rectangle((x, y), 0.98*self.actions[self.u], 0.1,
        #                       color=color, fill=True, lw=2))

        # Save figure
        filename = self.path+"/gif/"+str(self.stp_plot)+".png"
        plt.savefig(filename, dpi=100, bbox_inches="tight")
        plt.close()

        # Dump
        if dump: self.dump(self.path+"/vortex.dat")

        self.stp_plot += 1

    # Dump x, y, z
    def dump(self, filename):

        array = self.ht.copy()
        array = np.vstack((array, self.hx[:,0]))
        array = np.vstack((array, self.hx[:,1]))
        array = np.vstack((array, self.hx[:,2]))
        array = np.vstack((array, self.hx[:,3]))
        array = np.transpose(array)

        np.savetxt(filename, array, fmt='%.5e')

    # Close environment
    def close(self):
        pass

###############################################
### Five-stage fourth-order low-storage Runge-Kutta class
class lsrk4():
    def __init__(self):

        # lsrk coefficients
        self.n_lsrk = 5
        self.a = np.array([ 0.000000000000000, -0.417890474499852,
                           -1.192151694642677, -1.697784692471528,
                           -1.514183444257156])
        self.b = np.array([ 0.149659021999229,  0.379210312999627,
                            0.822955029386982,  0.699450455949122,
                            0.153057247968152])
        self.c = np.array([ 0.000000000000000,  0.149659021999229,
                            0.370400957364205,  0.622255763134443,
                            0.958282130674690])

    # return number of integration steps
    def steps(self):

        return self.n_lsrk

    # return source time at jth step
    def source_time(self, j, t, dt):

        return t + self.c[j]*dt

    # lsrk update
    def update(self, u, uk, f, j, dt):

        for i in range(len(u)):
            u[i]   = self.a[j]*u[i] + dt*f[i]
            uk[i] += self.b[j]*u[i]
