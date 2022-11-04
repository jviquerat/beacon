### Generic imports
import os
import time
import gym
import gym.spaces        as gsp
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

from   numba              import jit, njit
from   matplotlib.patches import Rectangle

###############################################
### Generic class
class shkadov(gym.Env):
    metadata = {'render.modes': ['human']}

    # Initialize instance
    def __init__(self, cpu=0, n_jets=1, jet_pos=150.0, jet_space=25.0):

        # Main parameters
        self.L          = 300        # length of domain in mm
        self.nx         = 600        # nb of discretization points
        self.dt         = 0.005      # timestep
        self.dt_act     = 0.05       # action timestep
        self.t_warmup   = 100.0      # warmup time
        self.t_act      = 20.0       # action time after warmup
        self.sigma      = 5.0e-4     # input noise
        self.n_jets     = n_jets     # nb of jets
        self.jet_amp    = 5.0        # jet amplitude scaling
        self.jet_pos    = jet_pos    # position of first jet
        self.jet_hw     = 1.5        # jet half-width
        self.jet_space  = jet_space  # spacing between jets
        self.l_obs      = 25.0       # length for upstream observations
        self.l_rwd      = 10.0       # length for downstream reward
        self.u_interp   = 0.02       # time on which action is interpolated
        self.blowup_rwd =-10.0       # reward in case of blow-up
        self.init_file  = "init.dat" # initialization file

        # Deduced parameters
        self.t_max      = self.t_warmup + self.t_act     # total simulation time
        self.dx         = float(self.L/self.nx)          # spatial step
        self.ndt_max    = int(self.t_max/self.dt)        # nb of numerical timesteps
        self.ndt_act    = int(self.dt_act/self.dt)       # nb of numerical timesteps per action
        self.ndt_warmup = int(self.t_warmup/self.dt)     # nb of numerical timesteps for warmup
        self.n_act      = int(self.t_act/self.dt_act)    # nb of action steps per episode
        self.n_warmup   = int(self.t_warmup/self.dt_act) # nb of action steps for warmup
        self.n_interp   = int(self.u_interp/self.dt)     # nb of interpolation steps for action
        self.jet_pos    = int(self.jet_pos/self.dx)      # jet position in lattice units
        self.jet_hw     = int(self.jet_hw/self.dx)       # jet half-width in lattice units
        self.jet_space  = int(self.jet_space/self.dx)    # jet spacing in lattice units
        self.jet_start  = self.jet_pos - self.jet_hw     # jet starting index in lattice units
        self.jet_end    = self.jet_pos + self.jet_hw     # jet ending index in lattice units
        self.l_rwd      = int(self.l_rwd/self.dx)        # length of rwd zone in lattice units
        self.l_obs      = int(self.l_obs/self.dx)        # length of obs zone in lattice units
        self.rwd_start  = self.jet_pos                   # start of rwd zone in lattice units
        self.rwd_end    = self.jet_pos + self.l_rwd      # end of rwd zone in lattice units
        self.obs_start  = self.jet_pos - self.l_obs      # start of obs zone in lattice units
        self.obs_end    = self.jet_pos                   # end of obs zone in lattice units
        self.n_obs      = self.obs_end - self.obs_start  # nb of lattice points in obs zone

        ### Path
        self.path = "png"
        os.makedirs(self.path, exist_ok=True)

        ### Declare arrays
        self.x       = np.linspace(0, self.nx, num=self.nx, endpoint=False)*self.dx
        self.h       = np.zeros((self.nx)) # current h
        self.q       = np.zeros((self.nx)) # current q
        self.hp      = np.zeros((self.nx)) # previous h
        self.qp      = np.zeros((self.nx)) # previous q
        self.q2h     = np.zeros((self.nx)) # current q**2/h
        self.dq2h    = np.zeros((self.nx)) # 1st-order derivative of q**2/h
        self.dddh    = np.zeros((self.nx)) # 3rd-order derivative of h
        self.ddh     = np.zeros((self.nx)) # 2nd-order derivative of h
        self.dq      = np.zeros((self.nx)) # 1st-order derivative of q
        self.rhsq    = np.zeros((self.nx)) # 3rd-order derivative of h
        self.h_init  = np.zeros((self.nx)) # h initialization
        self.q_init  = np.zeros((self.nx)) # q initialization
        self.hp_init = np.zeros((self.nx)) # hp initialization
        self.qp_init = np.zeros((self.nx)) # qp initialization

        # Load initialization file
        self.load(self.init_file)

        # Define action space
        self.action_space = gsp.Box(low   =-1.0,
                                    high  = 1.0,
                                    shape = (self.n_jets,),
                                    dtype = np.float32)

        # Define observation space
        self.h_max = 3.0
        self.q_max = 5.0
        high = np.array([])
        high_h = np.ones((self.n_obs))*self.h_max
        high_q = np.ones((self.n_obs))*self.q_max
        for i in range(self.n_jets):
            high = np.append(high, high_h)
            high = np.append(high, high_q)
        self.observation_space = gsp.Box(low   =-high,
                                         high  = high,
                                         shape = (2*self.n_obs*self.n_jets,),
                                         dtype = np.float32)

    # Reset environment
    def reset(self):

        self.reset_fields()
        self.h[:]  = self.h_init[:]
        self.hp[:] = self.hp_init[:]
        self.q[:]  = self.q_init[:]
        self.qp[:] = self.qp_init[:]

        obs = self.get_obs()

        return obs, None

    # Reset fields to initial state
    def reset_fields(self):

        # Initial solution
        self.h[:]  = 1.0
        self.q[:]  = 1.0
        self.hp[:] = 1.0
        self.qp[:] = 1.0

        # Other fields
        self.q2h[:]  = 0.0
        self.dq2h[:] = 0.0
        self.dddh[:] = 0.0
        self.ddh[:]  = 0.0
        self.dq[:]   = 0.0
        self.rhsq[:] = 0.0

        # Actions
        self.u     = [0.0]*self.n_jets
        self.u_prv = [0.0]*self.n_jets

        # Running indices
        self.stp      = 0
        self.stp_plot = 0

    # Run warmup
    def warmup(self):

        # Run until flow is developed
        for i in range(self.n_warmup):
            self.solve()

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
        if (np.any((self.h < -self.h_max) | (self.h > self.h_max))):
            print("Blowup")
            done  = True
            trunc = False
            rwd   = self.blowup_rwd

        # Update step
        self.stp += 1

        return obs, rwd, done, trunc, None

    # Resolution
    def solve(self, u=None):

        if (u is None): u = self.u.copy()

        # Save actions
        self.u_prv[:] = self.u[:]
        self.u[:]     = u[:]

        # Run solver
        for i in range(self.ndt_act):

            # Update previous fields
            self.hp[:] = self.h[:]
            self.qp[:] = self.q[:]

            # Boundary conditions
            self.h[0]    = 1.0 + np.random.uniform(-self.sigma, self.sigma, 1)
            self.q[0]    = 1.0
            self.h[self.nx-1] = self.h[self.nx-2]
            self.q[self.nx-1] = self.q[self.nx-2]

            # Compute q2h and first spatial derivative
            self.q2h[:] = self.q[:]*self.q[:]/self.h[:]
            d1tvd(self.q2h, self.dq2h, self.nx, self.dx)

            # Compute q first spatial derivative
            d1tvd(self.q, self.dq, self.nx, self.dx)

            # Add control
            alpha = min(float(i)/float(self.n_interp), 1.0)
            for j in range(self.n_jets):
                u[j] = (1.0-alpha)*self.u_prv[j] + alpha*self.u[j]
                s = self.jet_pos + j*self.jet_space - self.jet_hw
                e = s + 2*self.jet_hw + 1
                for k in range(s,e):
                    self.dq[k] += self.jet_amp*u[j]/(self.jet_hw**2)

            # Compute h third spatial derivative
            d2o2c(self.h, self.ddh, self.nx, self.dx)
            d1o2u(self.ddh, self.dddh, self.nx, self.dx)

            # Compute rhs
            rhs(self.dq2h, self.h, self.dddh, self.q, self.rhsq, self.nx)

            # March in time
            update_o1(self.h, self.hp, self.dq, self.nx, self.dt)
            update_o1(self.q, self.qp, self.rhsq, self.nx, self.dt)

    # Retrieve observations
    def get_obs(self):

        obs = np.array([])
        tmp = np.zeros((2*self.n_obs))
        for i in range(self.n_jets):
            s = self.jet_pos + i*self.jet_space - self.l_obs
            e = s + self.l_obs
            tmp[:self.n_obs] = self.h[s:e]
            tmp[self.n_obs:] = self.q[s:e]
            obs = np.append(obs, tmp)

        return obs

    # Compute reward
    def get_rwd(self):

        rwd   = 0.0
        hdiff = np.zeros((self.l_rwd))
        for i in range(self.n_jets):
            s = self.jet_pos + i*self.jet_space
            e = s + self.l_rwd
            hdiff[:]  = self.h[s:e]
            hdiff[:] -= 1.0
            rwd      -= np.sum(np.square(hdiff))/(e-s)
        rwd /= self.n_jets

        return rwd

    # Render environment
    def render(self, mode="human", show=False):

        ### Initialize plot
        if (self.stp_plot == 0):
            plt.figure(figsize=(10,2))

        ax  = plt.gca()
        fig = plt.gcf()
        ax.set_xlim([0.0,self.L])
        ax.set_ylim([0.0,2.0])
        plt.plot(self.x,self.h)
        for i in range(self.n_jets):
            s = self.jet_pos + i*self.jet_space - self.jet_hw
            ax.add_patch(Rectangle((s*self.dx, 1.0),
                                   2.0*self.jet_hw*self.dx, self.u[i],
                                   facecolor='red', fill=True, lw=1))
        fig.tight_layout()
        plt.grid()
        fig.savefig(self.path+'/'+str(self.stp_plot)+'.png',
                    bbox_inches='tight')
        if show: plt.pause(0.0001)
        plt.clf()
        self.stp_plot += 1

    # Dump (h,q)
    def dump(self, filename):

        array = self.h.copy()
        array = np.vstack((array, self.hp))
        array = np.vstack((array, self.q))
        array = np.vstack((array, self.qp))
        array = np.transpose(array)

        np.savetxt(filename, array, fmt='%.5e')

    # Load (h,q)
    def load(self, filename):

        f = np.loadtxt(filename)
        self.h_init[:]  = f[:,0]
        self.hp_init[:] = f[:,1]
        self.q_init[:]  = f[:,2]
        self.qp_init[:] = f[:,3]

    # Closing
    def close(self):
        pass

###############################################
# 1st derivative, 1st order upstream
@jit(cache=True,fastmath=True)
def d1o1u(u, du, nx, dx):

    du[1:nx] = (u[1:nx] - u[0:nx-1])/dx

# 2nd derivative, 2nd order centered
@jit(cache=True,fastmath=True)
def d2o2c(u, du, nx, dx):

    du[1:nx-1] = (u[2:nx] - 2.0*u[1:nx-1] + u[0:nx-2])/(dx*dx)

# 1st derivative, 2nd order upstream
@jit(cache=True,fastmath=True)
def d1o2u(u, du, nx, dx):

    du[0:nx-2] = (-u[2:nx] + 4.0*u[1:nx-1] - 3.0*u[0:nx-2])/(2.0*dx)
    du[nx-2]   = (u[nx-1] - u[nx-2])/dx

# 1st derivative, minmod tvd scheme
@jit(cache=True,fastmath=True)
def d1tvd(u, du, nx, dx):

    phi         = np.zeros((nx))
    r           = np.zeros((nx))
    r[1:nx-1]   = (u[1:nx-1] - u[0:nx-2])/(u[2:nx] - u[1:nx-1] + 1.0e-8)
    phi[1:nx-1] = np.maximum(0.0, np.minimum(r[1:nx-1], 1.0)) # mindmod
    #phi[1:nx-1] = (r[1:nx-1] + np.absolute(r[1:nx-1]))/(1.0 + r[1:nx-1]) # van-leer

    du[1:nx-1]  = u[1:nx-1] + 0.5*phi[1:nx-1]*(u[2:nx]   - u[1:nx-1])
    du[1:nx-1] -= u[0:nx-2] + 0.5*phi[0:nx-2]*(u[1:nx-1] - u[0:nx-2])
    du[1:nx-1] /= dx

# 1st order update in time
@jit(cache=True,fastmath=True)
def update_o1(u, up, rhs, nx, dt):

    u[1:nx-1] = up[1:nx-1] - dt*rhs[1:nx-1]

# 2nd order update in time
@jit(cache=True,fastmath=True)
def update_o2(u, up, upp, rhs, nx, dt):

    u[1:nx-1] = (4.0*up[1:nx-1] - upp[1:nx-1] - 2.0*dt*rhs[1:nx-1])/3.0

# rhs computation
@jit(cache=True,fastmath=True)
def rhs(dq2h, h, dddh, q, rhsq, nx):
    rhsq[1:nx-1] = 1.2*dq2h[1:nx-1] - 2.0*(h[1:nx-1]*(dddh[1:nx-1] + 1.0)
                                           - q[1:nx-1]/(h[1:nx-1]*h[1:nx-1]))
0
