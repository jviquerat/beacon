### Generic imports
import os
import time
import math
import random
import gym
import gym.spaces        as gsp
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

from   matplotlib.patches import Rectangle

###############################################
### Lorenz environment
class lorenz(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, init=True,
                 sigma=10.0, rho=28.0, beta=8.0/3.0):

        # Main parameters
        self.dt          = 0.01   # timestep
        self.dt_act      = 0.05   # action timestep
        self.sigma       = sigma  # lorenz parameter
        self.rho         = rho    # lorenz parameter
        self.beta        = beta   # lorenz parameter
        self.t_max       = 20.0   # total simulation time

        # Deduced parameters
        self.n_obs      = 6                           # total nb of observations
        self.ndt_max    = int(self.t_max/self.dt)     # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)    # nb of numerical timesteps per action
        self.t_act      = self.t_max                  # action time
        self.n_act      = int(self.t_act/self.dt_act) # nb of action steps per episode

        #self.gif_steps = 300

        ### Path
        self.path = "png"
        os.makedirs(self.path, exist_ok=True)

        # Arrays
        self.x  = np.zeros(3)                 # unknowns
        self.xk = np.zeros(3)                 # lsrk storage
        self.fx = np.zeros(3)                 # rhs
        self.hx = np.zeros((self.ndt_max, 3)) # time storage
        self.t  = np.linspace(0.0, self.t_max, num=self.ndt_max)

        # Initialize integrator
        self.integrator = lsrk4()

        # Define action space
        self.action_space = gsp.Discrete(3)
        self.actions = np.array([-1.0, 0.0, 1.0])

        # Define observation space
        high = np.ones((self.n_obs))/40.0
        self.observation_space = gsp.Box(low   =-high,
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
        self.x[0] = 10.0
        self.x[1] = 10.0
        self.x[2] = 10.0

        # Other fields
        self.xk[:] = 0.0
        self.fx[:] = 0.0
        self.hx[:] = 0.0

        # Actions
        self.u  = 0
        self.up = 0

        # Observations
        self.obs = np.zeros((2, 3))

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Step
    def step(self, u=None):

        # Copy first step
        self.hx[0,:] = self.x[:]

        # Run solver
        self.solve(self.actions[u])

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
        self.up = self.u
        self.u  = u

        # Run solver
        for i in range(self.ndt_act):

            # Update intermediate storage
            self.xk[:] = self.x[:]

            # Loop on integration steps
            for j in range(self.integrator.steps()):

                # Compute rhs without forcing term
                self.fx[0] = self.sigma*(self.xk[1] - self.xk[0])
                self.fx[1] = self.xk[0]*(self.rho - self.xk[2]) - self.xk[1]
                self.fx[2] = self.xk[0]*self.xk[1] - self.beta*self.xk[2]

                # Add forcing term
                self.fx[1] += u

                # Update
                self.integrator.update(self.x, self.xk, self.fx, j, self.dt)

            # Update main storage
            self.x[:] = self.xk[:]

            # Store unknowns
            self.hx[self.stp*self.ndt_act+i,:] = self.x[:]

    # Retrieve observations
    def get_obs(self):

        # Fill new observations
        self.obs[0,:] = self.x[:]
        self.obs[1,:] = self.fx[:]

        obs = np.reshape(self.obs, [-1])

        return obs

    # Compute reward
    def get_rwd(self):

        if (self.x[0] < 0.0): rwd = 1.0
        else: rwd = 0.0

        return rwd

    # Rendering
    def render(self, mode="human", show=False, dump=True):

        if (self.stp == self.n_act):
            plt.clf()
            plt.cla()
            fig, ax = plt.subplots(figsize=(8,2))
            fig.tight_layout()
            plt.plot(self.t, self.hx[:,0])
            ax.set_xlim([0.0, self.t_max])
            ax.set_ylim([-20.0, 20.0])

            filename = self.path+"/history.png"
            plt.grid()
            plt.savefig(filename, dpi=100)
            plt.close()

        if (self.stp_plot == 0):
            os.makedirs(self.path+"/gif", exist_ok=True)

        plt.clf()
        plt.cla()
        fig = plt.figure(tight_layout=True)
        ax  = fig.add_subplot(projection='3d')
        ax.set_axis_off()
        ax.set_xlim([-20.0, 20.0])
        ax.set_ylim([-20.0, 20.0])
        ax.set_zlim([  0.0, 40.0])
        ax.plot(self.hx[:self.stp,0],
                self.hx[:self.stp,1],
                self.hx[:self.stp,2],
                linewidth=1)

        filename = self.path+"/gif/"+str(self.stp_plot)+".png"
        bbox = fig.bbox_inches.from_bounds(1, 1.25, 4.75, 3)
        plt.savefig(filename, dpi=100, bbox_inches=bbox)
        plt.close()

        if dump: self.dump(self.path+"/lorenz.dat")

        self.stp_plot += 1

    # Dump x, y, z
    def dump(self, filename):

        array = self.t.copy()
        array = np.vstack((array, self.hx[:,0]))
        array = np.vstack((array, self.hx[:,1]))
        array = np.vstack((array, self.hx[:,2]))
        array = np.transpose(array)

        np.savetxt(filename, array, fmt='%.5e')

    # # Rendering with gif
    # def render_gif(self, x):

    #     os.makedirs(self.path+"/lorenz_gif", exist_ok=True)

    #     stp = math.floor(self.n_steps/self.gif_steps)
    #     for i in range(self.gif_steps):
    #         plt.clf()
    #         plt.cla()
    #         fig = plt.figure(tight_layout=True)
    #         ax  = fig.add_subplot(projection='3d')
    #         ax.set_axis_off()
    #         ax.set_xlim([-20.0, 20.0])
    #         ax.set_ylim([-20.0, 20.0])
    #         ax.set_zlim([  0.0, 40.0])
    #         j = i*stp
    #         ax.plot(self.hx[:j,0], self.hx[:j,1], self.hx[:j,2],
    #                 linewidth=1)

    #         filename = self.path+"/lorenz_gif/"+str(i)+".png"
    #         bbox = fig.bbox_inches.from_bounds(1, 1.25, 4.75, 3)
    #         plt.savefig(filename, dpi=100, bbox_inches=bbox)
    #         plt.close()

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
