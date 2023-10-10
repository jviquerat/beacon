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
### Generic class
class mixing(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0,
                 L=1.0, H=1.0, re=100.0, pe=10000.0, C0=2.0):

        # Main parameters
        self.L           = L                # length of the domain
        self.H           = H                # height of the domain
        self.nx          = 100*int(self.L)  # nb of pts in x direction
        self.ny          = 100*int(self.H)  # nb of pts in y direction
        self.re          = re               # reynolds number
        self.pe          = pe               # peclet number
        self.C0          = C0               # initial concentration in patch
        self.side        = 0.5              # side size of initial concentration patch
        self.nu          = 0.01             # dynamic viscosity
        self.u_max       = re*self.nu/L     # max velocity
        self.dt          = 0.002            # timestep
        self.dt_act      = 0.5              # action timestep
        self.t_act       = 40.0             # action time after warmup
        self.nx_obs_pts  = 4*int(self.L)    # nb of obs pts in x direction
        self.ny_obs_pts  = 4*int(self.H)    # nb of obs pts in y direction
        self.n_obs_steps = 4                # nb of observations steps

        # Deduced parameters
        self.dx         = float(self.L/self.nx)           # x spatial step
        self.dy         = float(self.H/self.ny)           # y spatial step
        self.ndt_act    = int(self.dt_act/self.dt)        # nb of numerical timesteps per action
        self.n_act      = int(self.t_act/self.dt_act)     # nb of action steps per episode
        self.n_obs_tot  = 3*self.n_obs_steps*self.nx_obs_pts*self.ny_obs_pts # total number of observation pts
        self.nx_obs     = self.nx//self.nx_obs_pts        # nb of pts between each observation pt in x direction
        self.ny_obs     = self.ny//self.ny_obs_pts        # nb of pts between each observation pt

        ### Path
        self.path               = "png"
        self.concentration_path = self.path+"/concentration"
        self.field_path         = self.path+"/field"
        os.makedirs(self.path,               exist_ok=True)
        os.makedirs(self.concentration_path, exist_ok=True)
        os.makedirs(self.field_path,         exist_ok=True)

        ### Declare arrays
        self.u       = np.zeros((self.nx+2, self.ny+2)) # u field
        self.v       = np.zeros((self.nx+2, self.ny+2)) # v field
        self.p       = np.zeros((self.nx+2, self.ny+2)) # p field
        self.C       = np.zeros((self.nx+2, self.ny+2)) # C field
        self.us      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.vs      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.phi     = np.zeros((self.nx+2, self.ny+2)) # projection field

        # Define action space
        self.action_space = gsp.Discrete(2)
        self.actions = np.array([-self.u_max, self.u_max])
        # self.action_space = gsp.Box(low   =-self.u_max,
        #                             high  = self.u_max,
        #                             shape = (2,),
        #                             dtype = np.float32)

        # Define observation space
        high = np.ones(self.n_obs_tot)*self.u_max

        self.observation_space = gsp.Box(low   =-high,
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
        self.u[:,:] = 0.0
        self.v[:,:] = 0.0
        self.p[:,:] = 0.0
        self.C[:,:] = 0.0

        i_min = math.floor(0.5*(self.L-self.side)/self.dx)
        i_max = i_min + math.floor(self.side/self.dx)
        j_min = math.floor(0.5*(self.H-self.side)/self.dy)
        j_max = j_min + math.floor(self.side/self.dy)
        self.C[i_min:i_max,j_min:j_max] = self.C0

        # Other fields
        self.us[:,:]  = 0.0
        self.vs[:,:]  = 0.0
        self.phi[:,:] = 0.0

        # Actions
        self.a  = 1
        self.ap = 1

        # Observations
        self.obs = np.zeros((self.n_obs_steps, 3,
                             self.nx_obs_pts,
                             self.ny_obs_pts))

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

        if (a == 0):
            v_lat = 0.0
            u_top = self.u_max
        if (a == 1):
            v_lat = self.u_max
            u_top = 0.0

        # Run solver
        for i in range(self.ndt_act):

            #########################
            # Set boundary conditions
            #########################

            # Left wall
            self.u[1,1:-1]  = 0.0
            self.v[0,2:-1]  =-2.0*v_lat - self.v[1,2:-1]
            self.C[0,1:-1]  = self.C[1,1:-1]

            # Right wall
            self.u[-1,1:-1] = 0.0
            self.v[-1,2:-1] = 2.0*v_lat - self.v[-2,2:-1]
            self.C[-1,1:-1] = self.C[-2,1:-1]

            # Top wall
            self.u[1:,-1]   = 2.0*u_top - self.u[1:,-2]
            self.v[1:-1,-1] = 0.0
            self.C[1:-1,-1] = self.C[1:-1,-2]

            # Bottom wall
            self.u[1:,0]    =-2.0*u_top - self.u[1:,1]
            self.v[1:-1,1]  = 0.0
            self.C[1:-1,0]  = self.C[1:-1,1]

            #########################
            # Predictor step
            # Computes starred fields
            #########################

            predictor(self.u, self.v, self.us, self.vs, self.p,
                      self.nx, self.ny, self.dt, self.dx, self.dy, self.re)

            #########################
            # Poisson step
            # Computes pressure field
            #########################

            itp, ovf = poisson(self.us, self.vs, self.u, self.phi,
                               self.nx, self.ny, self.dx, self.dy, self.dt)
            self.p[:,:] += self.phi[:,:]

            if (ovf):
                print("\n")
                print("Exceeded max number of iterations in solver")
                exit(1)

            #########################
            # Corrector step
            # Computes div-free fields
            #########################

            corrector(self.u, self.v, self.us, self.vs, self.phi,
                      self.nx, self.ny, self.dx, self.dy, self.dt)

            #########################
            # Transport step
            # Computes temperature field
            #########################

            transport(self.u, self.v, self.C,
                      self.nx, self.ny, self.dx, self.dy, self.dt, self.pe)

    # Retrieve observations
    def get_obs(self):

        # Copy previous observations
        for i in range(1,self.n_obs_steps):
            self.obs[i-1,:,:,:] = self.obs[i,:,:,:]

        # Fill new observations
        x = self.nx_obs//2
        for i in range(self.nx_obs_pts):
            y = self.ny_obs//2
            for j in range(self.ny_obs_pts):
                self.obs[-1,0,i,j] = self.C[x,y]
                self.obs[-1,1,i,j] = self.u[x,y]
                self.obs[-1,2,i,j] = self.v[x,y]
                y               += self.ny_obs
            x += self.nx_obs

        obs = np.reshape(self.obs, [-1])

        return obs

    # Compute reward
    def get_rwd(self):

        ref_c = (self.side*self.side)/(self.L*self.H)*self.C0
        r     =-np.mean(np.square(np.reshape(self.C, (-1)) - ref_c))

        return r

    # Render environment
    def render(self, mode="human", show=False, dump=True):

        # Set field
        pC      = np.zeros((self.nx, self.ny))
        pC[:,:] = self.C[1:-1,1:-1]

        # Rotate field
        pC = np.rot90(pC)

        # Plot temperature
        fig = plt.figure(figsize=(5,5.5))
        ax  = fig.add_subplot(30, 1, (1,28))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(pC,
                  cmap = 'RdBu_r',
                  vmin = 0.0,
                  vmax = self.C0,
                  extent=[0.0, self.L, 0.0, self.H])

        # Plot control
        ax = fig.add_subplot(30, 1, (29,30))
        ax.set_xlim([-self.u_max, self.u_max])
        ax.set_ylim([ 0.0, 0.1])
        ax.set_xticks([])
        ax.set_yticks([])
        x = 0.0
        y = 0.02
        color = 'r' if self.a > 0.0 else 'b'
        ax.add_patch(Rectangle((x,y), 0.98*self.actions[self.a], 0.06,
                               color=color, fill=True, lw=2))

        # Save figure
        filename = self.concentration_path+"/"+str(self.stp_plot)+".png"
        fig.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')

        # Show and dump
        if show: plt.pause(0.0001)
        if dump: self.dump(self.field_path+"/field_"+str(self.stp_plot)+".dat")

        self.stp_plot += 1

    # Dump fields
    def dump(self, field_name):

        array = self.u.copy()
        array = np.vstack((array, self.v))
        array = np.vstack((array, self.p))
        array = np.vstack((array, self.C))

        np.savetxt(field_name,   array,   fmt='%.5e')

    # Closing
    def close(self):
        pass

###############################################
# Predictor step
@nb.njit(cache=False)
def predictor(u, v, us, vs, p, nx, ny, dt, dx, dy, re):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            uE = 0.5*(u[i+1,j] + u[i,j])
            uW = 0.5*(u[i,j]   + u[i-1,j])
            uN = 0.5*(u[i,j+1] + u[i,j])
            uS = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j]   + v[i-1,j])
            conv = (uE*uE-uW*uW)/dx + (uN*vN-uS*vS)/dy

            diff  = ((u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/(dx**2) +
                     (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/(dy**2))/re

            pres = (p[i,j] - p[i-1,j])/dx

            us[i,j] = u[i,j] + dt*(diff - conv - pres)

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            vE = 0.5*(v[i+1,j] + v[i,j])
            vW = 0.5*(v[i,j]   + v[i-1,j])
            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j]   + u[i,j-1])
            vN = 0.5*(v[i,j+1] + v[i,j])
            vS = 0.5*(v[i,j]   + v[i,j-1])
            conv = (uE*vE-uW*vW)/dx + (vN*vN-vS*vS)/dy

            diff  = ((v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/(dx**2) +
                     (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/(dy**2))/re

            pres  = (p[i,j] - p[i,j-1])/dy

            vs[i,j] = v[i,j] + dt*(diff - conv - pres)

###############################################
# Poisson step
@nb.njit(cache=False)
def poisson(us, vs, u, phi, nx, ny, dx, dy, dt):

    tol      = 1.0e-4
    err      = 1.0e10
    itp      = 0
    itmax    = 300000
    ovf      = False
    phi[:,:] = 0.0
    phin     = np.zeros((nx+2,ny+2))
    while(err > tol):

        phin[:,:] = phi[:,:]

        for i in range(1,nx+1):
            for j in range(1,ny+1):

                b = ((us[i+1,j] - us[i,j])/dx +
                     (vs[i,j+1] - vs[i,j])/dy)/dt

                phi[i,j] = 0.5*((phin[i+1,j] + phin[i-1,j])*dy*dy +
                                (phin[i,j+1] + phin[i,j-1])*dx*dx -
                                b*dx*dx*dy*dy)/(dx*dx+dy*dy)

        # Domain left (neumann)
        phi[ 0,1:-1] = phi[ 1,1:-1]

        # Domain right (neumann)
        phi[-1,1:-1] = phi[-2,1:-1]

        # Domain top (dirichlet)
        phi[1:-1,-1] = 0.0

        # Domain bottom (neumann)
        phi[1:-1, 0] = phi[1:-1, 1]

        # Compute error
        dphi = np.reshape(phi - phin, (-1))
        err  = np.dot(dphi,dphi)

        itp += 1
        if (itp > itmax):
            ovf = True
            break

    return itp, ovf

###############################################
# Corrector step
@nb.njit(cache=False)
def corrector(u, v, us, vs, phi, nx, ny, dx, dy, dt):

    u[2:-1,1:-1] = us[2:-1,1:-1] - dt*(phi[2:-1,1:-1] - phi[1:-2,1:-1])/dx
    v[1:-1,2:-1] = vs[1:-1,2:-1] - dt*(phi[1:-1,2:-1] - phi[1:-1,1:-2])/dy

###############################################
# Transport step
@nb.njit(cache=False)
def transport(u, v, C, nx, ny, dx, dy, dt, pe):

    for i in range(1,nx+1):
        for j in range(1,ny+1):
            uE = u[i+1,j]
            uW = u[i,j]
            vN = v[i,j+1]
            vS = v[i,j]
            CE = 0.5*(C[i+1,j] + C[i,j])
            CW = 0.5*(C[i-1,j] + C[i,j])
            CN = 0.5*(C[i,j+1] + C[i,j])
            CS = 0.5*(C[i,j-1] + C[i,j])
            conv = (uE*CE-uW*CW)/dx + (vN*CN-vS*CS)/dy

            diff  = ((C[i+1,j] - 2.0*C[i,j] + C[i-1,j])/(dx**2) +
                     (C[i,j+1] - 2.0*C[i,j] + C[i,j-1])/(dy**2))/pe

            C[i,j] += dt*(diff - conv)
