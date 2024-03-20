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
class rayleigh(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, init=True,
                 L=1.0, H=1.0, n_sgts=10, ra=1.0e4):

        # Main parameters
        self.L           = L                # length of the domain
        self.H           = H                # height of the domain
        self.nx          = int(50*self.L)   # nb of pts in x direction
        self.ny          = int(50*self.H)   # nb of pts in y direction
        self.ra          = ra               # rayleigh number
        self.pr          = 0.71             # prandtl number
        self.Tc          =-0.5              # top plate temperature
        self.Th          = 0.5              # bottom plate reference temperature
        self.C           = 0.75             # max temperature variation at the bottom
        self.dt          = 0.01             # timestep
        self.dt_act      = 2.0              # action timestep
        self.t_warmup    = 200.0            # warmup time
        self.t_act       = 200.0            # action time after warmup
        self.n_sgts      = n_sgts           # nb of temperature segments
        self.nx_obs_pts  = 4*int(self.L)    # nb of obs pts in x direction
        self.ny_obs_pts  = 4*int(self.H)    # nb of obs pts in y direction
        self.n_obs_steps = 4                # nb of observations steps
        self.init_file   = "init_field.dat" # initialization file

        # Deduced parameters
        self.t_max      = self.t_warmup + self.t_act      # total simulation time
        self.dx         = float(self.L/self.nx)           # x spatial step
        self.dy         = float(self.H/self.ny)           # y spatial step
        self.ndt_max    = int(self.t_max/self.dt)         # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)        # nb of numerical timesteps per action
        self.ndt_warmup = int(self.t_warmup/self.dt)      # nb of numerical timesteps for warmup
        self.n_act      = int(self.t_act/self.dt_act)     # nb of action steps per episode
        self.n_warmup   = int(self.t_warmup/self.dt_act)  # nb of action steps for warmup
        self.nx_sgts    = self.nx//self.n_sgts            # nb of pts in each segment
        self.n_obs_tot  = 3*self.n_obs_steps*self.nx_obs_pts*self.ny_obs_pts # total number of observation pts
        self.nx_obs     = self.nx//self.nx_obs_pts        # nb of pts between each observation pt in x direction
        self.ny_obs     = self.ny//self.ny_obs_pts        # nb of pts between each observation pt

        # Declare arrays
        self.u       = np.zeros((self.nx+2, self.ny+2)) # u field
        self.v       = np.zeros((self.nx+2, self.ny+2)) # v field
        self.p       = np.zeros((self.nx+2, self.ny+2)) # p field
        self.T       = np.zeros((self.nx+2, self.ny+2)) # T field
        self.us      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.vs      = np.zeros((self.nx+2, self.ny+2)) # starred velocity field
        self.phi     = np.zeros((self.nx+2, self.ny+2)) # projection field

        self.u_init  = np.zeros((self.nx+2, self.ny+2)) # u initialization
        self.v_init  = np.zeros((self.nx+2, self.ny+2)) # v initialization
        self.p_init  = np.zeros((self.nx+2, self.ny+2)) # p initialization
        self.T_init  = np.zeros((self.nx+2, self.ny+2)) # T initializatino

        # Load initialization file
        if (init): self.load(self.init_file)

        # Define action space
        self.action_space = gsp.Box(low   =-self.C,
                                    high  = self.C,
                                    shape = (self.n_sgts,),
                                    dtype = np.float32)

        # Define observation space
        high = np.ones(self.n_obs_tot)

        self.observation_space = gsp.Box(low   =-high,
                                         high  = high,
                                         shape = (self.n_obs_tot,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()
        self.u[:] = self.u_init[:]
        self.v[:] = self.v_init[:]
        self.p[:] = self.p_init[:]
        self.T[:] = self.T_init[:]

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.u[:,:] = 0.0
        self.v[:,:] = 0.0
        self.p[:,:] = 0.0
        self.T[:,:] = 0.0

        # Other fields
        self.us[:,:]  = 0.0
        self.vs[:,:]  = 0.0
        self.phi[:,:] = 0.0

        # Actions
        self.a = [0.0]*self.n_sgts

        # Observations
        self.obs = np.zeros((self.n_obs_steps, 3,
                             self.nx_obs_pts,
                             self.ny_obs_pts))

        # Nusselt
        self.nu = np.empty((0,2))

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Run warmup
    def warmup(self):

        # Run until flow is developed
        for i in range(self.n_warmup):
            self.solve()

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

        # Zero-mean the actions
        a[:] = a[:] - np.mean(a)
        m    = max(1.0, np.amax(np.abs(a)/self.C))
        for i in range(self.n_sgts):
            a[i] = a[i]/m

        # Save actions
        self.a[:] = a[:]

        # Run solver
        for i in range(self.ndt_act):

            #########################
            # Set boundary conditions
            #########################

            # Left wall
            self.u[1,1:-1]  = 0.0
            self.v[0,2:-1]  =-self.v[1,2:-1]
            self.T[0,1:-1]  = self.T[1,1:-1]

            # Right wall
            self.u[-1,1:-1] = 0.0
            self.v[-1,2:-1] =-self.v[-2,2:-1]
            self.T[-1,1:-1] = self.T[-2,1:-1]

            # Top wall
            self.u[1:,-1]   =-self.u[1:,-2]
            self.v[1:-1,-1] = 0.0
            self.T[1:-1,-1] = 2.0*self.Tc - self.T[1:-1,-2]

            # Bottom wall
            self.u[1:,0]    =-self.u[1:,1]
            self.v[1:-1,1]  = 0.0

            for j in range(self.n_sgts):
                s = 1 + j*self.nx_sgts
                e = 1 + (j+1)*self.nx_sgts
                self.T[s:e,0]  = 2.0*(self.Th + self.a[j]) - self.T[s:e,1]

            #########################
            # Predictor step
            # Computes starred fields
            #########################

            predictor(self.u, self.v, self.us, self.vs, self.p, self.T,
                      self.nx, self.ny, self.dt, self.dx, self.dy, self.pr, self.ra)

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

            transport(self.u, self.v, self.T,
                      self.nx, self.ny, self.dx, self.dy, self.dt, self.pr, self.ra)

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
                self.obs[-1,0,i,j] = self.T[x,y]
                self.obs[-1,1,i,j] = self.u[x,y]
                self.obs[-1,2,i,j] = self.v[x,y]
                y               += self.ny_obs
            x += self.nx_obs

        obs = np.reshape(self.obs, [-1])

        return obs

    # Compute reward
    def get_rwd(self):

        nu = 0.0
        for i in range(1,self.nx+1):
            dT  = (self.T[i,1] - self.Th)/(0.5*self.dy)
            nu -= dT
        nu /= self.nx

        self.nu = np.append(self.nu, np.array([[self.stp, nu]]), axis=0)

        return -nu

    # Render environment
    def render(self, mode="human", show=False, dump=True):

        # Open directories
        if (self.stp_plot == 0):
            self.path             = "render"
            self.temperature_path = self.path+"/temperature"
            self.field_path       = self.path+"/field"
            self.action_path      = self.path+"/action"
            os.makedirs(self.path,             exist_ok=True)
            os.makedirs(self.temperature_path, exist_ok=True)
            os.makedirs(self.field_path,       exist_ok=True)
            os.makedirs(self.action_path,      exist_ok=True)

        # Set field
        pT      = np.zeros((self.nx, self.ny))
        pT[:,:] = self.T[1:-1,1:-1]

        # Rotate field
        pT = np.rot90(pT)

        # Plot temperature
        plt.clf()
        plt.cla()
        fig = plt.figure(figsize=(5,6))
        ax  = fig.add_subplot(30, 1, (1,29))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(pT,
                  cmap = 'RdBu_r',
                  vmin = self.Tc,
                  vmax = self.Th,
                  extent=[0.0, self.L, 0.0, self.H])

        # Plot control
        ax = fig.add_subplot(30, 1, (27,30))
        ax.set_xlim([0.0, self.n_sgts])
        ax.set_ylim([-self.C, self.C])
        ax.set_xticks([])
        ax.set_yticks([])

        ax.add_patch(Rectangle((0.0,0.0), self.n_sgts, 0.001,
                               color='k', fill=False, lw=0.3))
        w = 0.12
        for i in range(self.n_sgts):
            x = 0.5 + i - w
            y = 0.0
            color = 'r' if self.a[i] > 0.0 else 'b'
            ax.add_patch(Rectangle((x, y),
                                   2.0*w, 0.98*self.a[i]*self.C,
                                   color=color, fill=True, lw=1))

        # Save figure
        filename = self.temperature_path+"/"+str(self.stp_plot)+".png"
        fig.tight_layout()
        plt.savefig(filename, dpi=100, bbox_inches='tight')
        plt.close()

        # Show and dump
        if show: plt.pause(0.0001)
        if dump: self.dump(self.field_path+"/field_"+str(self.stp_plot)+".dat",
                           self.action_path+"/a_"+str(self.stp_plot)+".dat",
                           self.path+"/nu.dat")

        self.stp_plot += 1

    # Dump fields
    def dump(self, field_name, act_name, nusselt_name):

        array = self.u.copy()
        array = np.vstack((array, self.v))
        array = np.vstack((array, self.p))
        array = np.vstack((array, self.T))

        np.savetxt(field_name,   array,   fmt='%.5e')
        np.savetxt(act_name,     self.a,  fmt='%.5e')
        np.savetxt(nusselt_name, self.nu, fmt='%.5e')

    # Load (h,q)
    def load(self, filename):

        f = np.loadtxt(filename)
        self.u_init[:,:] = f[0*(self.nx+2):1*(self.nx+2),:]
        self.v_init[:,:] = f[1*(self.nx+2):2*(self.nx+2),:]
        self.p_init[:,:] = f[2*(self.nx+2):3*(self.nx+2),:]
        self.T_init[:,:] = f[3*(self.nx+2):4*(self.nx+2),:]

    # Closing
    def close(self):
        pass

###############################################
# Predictor step
@nb.njit(cache=False)
def predictor(u, v, us, vs, p, T, nx, ny, dt, dx, dy, pr, ra):

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
                     (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/(dy**2))
            diff *= math.sqrt(pr/ra)

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
                     (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/(dy**2))
            diff *= math.sqrt(pr/ra)

            pres  = (p[i,j] - p[i,j-1])/dy

            vs[i,j] = v[i,j] + dt*(diff - conv - pres + T[i,j])

###############################################
# Poisson step
@nb.njit(cache=False)
def poisson(us, vs, u, phi, nx, ny, dx, dy, dt):

    tol      = 1.0e-8
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
        phi[1:-1,-1] = phi[1:-1,-2]

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
def transport(u, v, T, nx, ny, dx, dy, dt, pr, ra):

    for i in range(1,nx+1):
        for j in range(1,ny+1):
            uE = u[i+1,j]
            uW = u[i,j]
            vN = v[i,j+1]
            vS = v[i,j]
            TE = 0.5*(T[i+1,j] + T[i,j])
            TW = 0.5*(T[i-1,j] + T[i,j])
            TN = 0.5*(T[i,j+1] + T[i,j])
            TS = 0.5*(T[i,j-1] + T[i,j])
            conv = (uE*TE-uW*TW)/dx + (vN*TN-vS*TS)/dy

            diff  = ((T[i+1,j] - 2.0*T[i,j] + T[i-1,j])/(dx**2) +
                     (T[i,j+1] - 2.0*T[i,j] + T[i,j-1])/(dy**2))
            diff /= math.sqrt(pr*ra)

            T[i,j] += dt*(diff - conv)
