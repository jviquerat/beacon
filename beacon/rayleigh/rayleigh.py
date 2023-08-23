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
                 L=1.0, H=1.0, n_sgts=5, ra=1.0e4):

        # Main parameters
        self.L          = L                 # length of the domain
        self.H          = H                 # height of the domain
        self.nx         = 100*int(self.L)   # nb of discretization points
        self.ra         = ra                # rayleigh number
        self.pr         = 0.71              # prandtl number
        self.g          = 9.81              # gravity
        self.Tc         =-0.5               # top plate temperature
        self.Th         = 0.5               # bottom plate reference temperature
        self.C          = 0.75              # max temperature variation at the bottom
        self.dt         = 0.0025            # timestep
        self.dt_act     = 0.05              # action timestep
        self.t_warmup   = 50.0              # warmup time
        self.t_act      = 20.0              # action time after warmup
        self.n_sgts     = n_sgts            # nb of temperature segments
        self.n_obs_pts  = 5                 # nb of obs pts per direction
        self.n_obs_tot  = self.n_obs_pts**2 # total number of observation pts
        self.u_interp   = 0.02              # time on which action is interpolated
        self.blowup_rwd =-1.0               # reward in case of blow-up
        self.eps        = 1.0e-8            # avoid division by zero
        self.init_file  = "init_field.dat"  # initialization file
        self.rand_init  = False             # random initialization
        self.rand_steps = 400               # nb of rand. steps for random initialization

        # Deduced parameters
        self.ny         = self.nx                        # nb of pts in y direction
        self.t_max      = self.t_warmup + self.t_act     # total simulation time
        self.dx         = float(self.L/self.nx)          # x spatial step
        self.dy         = self.dx                        # y spatial step
        self.ndt_max    = int(self.t_max/self.dt)        # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)       # nb of numerical timesteps per action
        self.ndt_warmup = int(self.t_warmup/self.dt)     # nb of numerical timesteps for warmup
        self.n_act      = int(self.t_act/self.dt_act)    # nb of action steps per episode
        self.n_warmup   = int(self.t_warmup/self.dt_act) # nb of action steps for warmup
        self.n_interp   = int(self.u_interp/self.dt)     # nb of interpolation steps for action

        ### Path
        self.path = "png"
        os.makedirs(self.path, exist_ok=True)

        ### Declare arrays
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
        low  = np.ones(self.n_obs_tot)*self.Tc
        high = np.ones(self.n_obs_tot)*(self.Th + self.C)

        self.observation_space = gsp.Box(low   = low,
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

        if (self.rand_init):
            n = random.randint(0,self.rand_steps)
            for i in range(n):
                self.step(self.a)
            self.stp = 0

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
        self.a  = [0.0]*self.n_sgts
        self.ap = [0.0]*self.n_sgts

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
        # if (np.any((self.h < -5.0*self.h_max) | (self.h > 5.0*self.h_max))):
        #     print("Blowup")
        #     done  = True
        #     trunc = False
        #     rwd   = self.blowup_rwd

        # Update step
        self.stp += 1

        return obs, rwd, done, trunc, None

    # Resolution
    def solve(self, a=None):

        if (a is None): a = self.a.copy()

        # Save actions
        self.ap[:] = self.a[:]
        self.a[:]  = a[:]

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
            self.T[1:-1,0]  = 2.0*self.Th - self.T[1:-1,1]

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

            # # Add control
            # alpha = min(float(i)/float(self.n_interp), 1.0)
            # for j in range(self.n_jets):
            #     u[j] = (1.0-alpha)*self.up[j] + alpha*self.u[j]
            #     s = self.jet_pos + j*self.jet_space - self.jet_hw
            #     e = s + 2*self.jet_hw
            #     for k in range(s,e+1):
            #         v  = (k-s)*(e-k)/(0.25*(e-s)**2)
            #         dq = self.jet_amp*u[j]*v
            #         self.rhsq[k] += dq

    # Retrieve observations
    def get_obs(self):

        obs = np.array([self.n_obs_pts, self.n_obs_pts])
        # for i in range(self.n_obs_pts):
        #     s = self.jet_pos + i*self.jet_space - self.l_obs
        #     e = s + self.l_obs
        #     stp = int(1.0/self.dx)
        #     tmp[:self.n_obs] = self.q[s:e:stp]
        #     obs = np.append(obs, tmp)

        return obs

    # Compute reward
    def get_rwd(self):

        nu = 0.0
        for i in range(1,self.nx+1):
            dT  = (self.T[i,1] - self.Th)/(0.5*self.dy)
            nu -= dT
        nu /= self.nx

        return -nu

    # Render environment
    def render(self, mode="human", show=False, dump=True):

        ### Initialize plot
        if (self.stp_plot == 0):
            plt.figure(figsize=(10,2))

        # Recreate fields at cells centers
        pu = np.zeros((self.nx, self.ny))
        pv = np.zeros((self.nx, self.ny))
        pT = np.zeros((self.nx, self.ny))

        pu[:,:] = 0.5*(self.u[2:,1:-1] + self.u[1:-1,1:-1])
        pv[:,:] = 0.5*(self.v[1:-1,2:] + self.v[1:-1,1:-1])
        pT[:,:] = self.T[1:-1,1:-1]

        # Compute velocity norm
        vn = np.sqrt(pu*pu+pv*pv)

        # Rotate fields
        vn = np.rot90(vn)
        pT = np.rot90(pT)

        # Create figure
        plt.clf()
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        fig.subplots_adjust(0.01, 0.01, 0.99, 0.99, wspace=0.02, hspace=0)

        # Plot velocity
        ax[0].axis('off')
        ax[0].imshow(vn,
                     cmap = 'RdBu_r',
                     vmin = 0.0,
                     vmax = 0.3)

        # Plot temperature
        ax[1].axis('off')
        ax[1].imshow(pT,
                     cmap = 'RdBu_r',
                     vmin = self.Tc,
                     vmax = self.Th)

        # Save figure
        filename = self.path+"/"+str(self.stp_plot)+".png"
        plt.savefig(filename, dpi=100)
        if show: plt.pause(0.0001)
        plt.close()

        # Dump if required
        if dump: self.dump(self.path+"/field_"+str(self.stp_plot)+".dat",
                           self.path+"/a_"+str(self.stp_plot)+".dat")

        self.stp_plot += 1

    # Dump fields
    def dump(self, field_name, act_name):

        array = self.u.copy()
        array = np.vstack((array, self.v))
        array = np.vstack((array, self.p))
        array = np.vstack((array, self.T))

        np.savetxt(field_name, array, fmt='%.5e')
        np.savetxt(act_name, self.a, fmt='%.5e')

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
