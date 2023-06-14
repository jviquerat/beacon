# Generic imports
import time
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

###############################################
### Cavity class
class cavity():

    ### Initialize
    def __init__(self, l=1.0, h=1.0, dx=0.01, dy=0.01, t_max=1.0, cfl=0.95, re=100.0):

        # Set parameters
        self.l     = l
        self.h     = h
        self.dx    = dx
        self.dy    = dy
        self.idx   = 1.0/dx
        self.idy   = 1.0/dy
        self.ifdxy = 0.5/(dx**2+dy**2)
        self.t_max = t_max
        self.cfl   = cfl
        self.nu    = 0.01
        self.re    = re
        self.utop  = self.re*self.nu/self.l

        # Compute nb of unknowns
        self.nx = round(self.l/self.dx)
        self.ny = round(self.h/self.dy)
        self.nc = self.nx*self.ny

        # Reset fields
        self.reset_fields()

        # Compute timestep
        self.tau = self.l/self.utop
        mdxy     = min(self.dx, self.dy)
        self.dt  = self.cfl*min(self.tau/self.re,
                                self.tau*self.re*mdxy**2/(4.0*self.l**2))

    ### Reset fields
    def reset_fields(self):

        # Set fields
        # Accound for boundary cells
        self.u  = np.zeros((self.nx+2, self.ny+2))
        self.v  = np.zeros((self.nx+2, self.ny+2))
        self.p  = np.zeros((self.nx+2, self.ny+2))
        self.us = np.zeros((self.nx+2, self.ny+2))
        self.vs = np.zeros((self.nx+2, self.ny+2))
        # self.u_old = np.zeros((self.nx+2, self.ny+2))
        # self.v_old = np.zeros((self.nx+2, self.ny+2))
        # self.p_old = np.zeros((self.nx+2, self.ny+2))
        # self.phi   = np.zeros((self.nx+2, self.ny+2))



        # Array to store iterations of poisson resolution
        self.n_itp = np.array([], dtype=np.int16)

        # Set time
        self.it = 0
        self.t  = 0.0

    ### Set boundary conditions
    def set_bc(self):

        # Left wall
        self.u[1,1:-1]  = 0.0
        self.v[0,1:]    =-self.v[1,1:]

        # Right wall
        self.u[-1,1:-1] = 0.0
        self.v[-1,1:]   = 0.0

        # Top wall
        self.u[1:,-1]   = 2.0*self.utop - self.u[1:,-2]
        self.v[1:-1,-1] = 0.0

        # Bottom wall
        self.u[1:,0]    =-self.u[1:,1]
        self.v[1:-1,1]  = 0.0

    ### Compute starred fields
    def predictor(self):

        predictor(self.u, self.v, self.us, self.vs, self.nx, self.ny,
                  self.dt, self.dx, self.dy, self.re)

    ### Compute pressure
    def poisson(self):

        itp, ovf = poisson(self.us, self.vs, self.p, self.nx, self.ny,
                           self.dx, self.dy, self.dt)

        self.n_itp = np.append(self.n_itp, np.array([self.it, itp]))

        # # Save previous pressure field
        # self.p_old[:,:] = self.p[:,:]

        # # Set pressure difference
        # self.phi[:,:] = 0.0

        # ovf, n_itp = poisson(self.us, self.vs, self.phi,
        #                      self.dx, self.dy, self.idx, self.idy,
        #                      self.nx, self.ny, self.dt,  self.ifdxy,
        #                      self.c_xmin, self.c_xmax, self.c_ymin, self.c_ymax)
        # self.n_itp = np.append(self.n_itp, np.array([self.it, n_itp]))

        # # Compute new pressure
        # self.p[:,:] = self.phi[:,:] + self.p_old[:,:]

        if (ovf):
            print("\n")
            print("Exceeded max number of iterations in solver")
            self.plot_fields()
            self.plot_iterations()
            exit(1)

    ### Compute updated fields
    def corrector(self):

        corrector (self.u, self.v, self.us, self.vs, self.p,
                   self.nx, self.ny, self.dx, self.dy, self.dt)

        #self.u[2:-1,1:-1] = self.us[2:-1,1:-1] - self.dt * (self.p[2:-1,1:-1] - self.p[1:-2,1:-1])/self.dx
        #self.v[1:-1,2:-1] = self.vs[1:-1,2:-1] - self.dt * (self.p[1:-1,2:-1] - self.p[1:-1,1:-2])/self.dy

        # self.u_old[:,:] = self.u[:,:]
        # self.v_old[:,:] = self.v[:,:]

        # corrector(self.u,   self.v,   self.us, self.vs, self.phi,
        #           self.idx, self.idy, self.nx, self.ny, self.dt)

    ### Take one step
    def step(self):

        self.set_bc()
        self.predictor()
        self.poisson()
        self.corrector()

        self.t  += self.dt
        self.it += 1

    ### Print
    def print(self):

        print('# t = '+str(self.t)+' / '+str(self.t_max), end='\r')

    ### Plot field norm
    def plot_fields(self):

        nx = self.nx
        ny = self.ny

        # Recreate fields at cells centers
        u = np.zeros((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))
        p = np.zeros((self.nx, self.ny))

        u[:,:] = 0.5*(self.u[2:,1:-1] + self.u[1:-1,1:-1])
        v[:,:] = 0.5*(self.v[1:-1,2:] + self.v[1:-1,1:-1])
        p[:,:] = self.p[1:-1,1:-1]

        # Compute velocity norm
        vn = np.sqrt(u*u+v*v)

        # Rotate fields
        vn = np.rot90(vn)
        p = np.rot90(p)

        # Plot velocity
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(vn,
                   cmap = 'RdBu_r',
                   vmin = 0.0,
                   vmax = self.utop)

        filename = "velocity.png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        # Plot pressure
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(p,
                   cmap = 'RdBu_r',
                   vmin =-2.0,
                   vmax = 4.0)

        filename = "pressure.png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

    ### Plot nb of solver iterations
    def plot_iterations(self):

        n_itp = np.reshape(self.n_itp, (-1,2))

        plt.clf()
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(n_itp[:,0], n_itp[:,1], color='blue')
        ax.grid(True)
        fig.tight_layout()
        filename = "iterations.png"
        fig.savefig(filename)
        np.savetxt("iterations.dat", self.n_itp, fmt='%d')

###############################################
# Predictor step
@nb.njit(cache=True)
def predictor(u, v, us, vs, nx, ny, dt, dx, dy, re):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            uE = 0.5*(u[i+1,j] + u[i,j])
            uW = 0.5*(u[i,j]   + u[i-1,j])

            uN = 0.5*(u[i,j+1] + u[i,j])
            uS = 0.5*(u[i,j] + u[i,j-1])

            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j] + v[i-1,j])

            conv = - (uE*uE-uW*uW)/dx - (uN*vN-uS*vS)/dy

            diff = ((u[i+1,j]-2.0*u[i,j]+u[i-1,j])/(dx**2) +
                    (u[i,j+1]-2.0*u[i,j]+u[i,j-1])/(dy**2))/re

            us[i,j] = u[i,j] + dt*(conv + diff)

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            vE = 0.5*(v[i+1,j] + v[i,j])
            vW = 0.5*(v[i,j] + v[i-1,j])

            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j] + u[i,j-1])

            vN = 0.5*(v[i,j+1] + v[i,j])
            vS = 0.5*(v[i,j] + v[i,j-1])

            conv = - (uE*vE-uW*vW)/dx - (vN*vN-vS*vS)/dy

            diff = ((v[i+1,j]-2.0*v[i,j]+v[i-1,j])/(dx**2) +
                    (v[i,j+1]-2.0*v[i,j]+v[i,j-1])/(dy**2))/re

            vs[i,j] = v[i,j] + dt*(conv + diff)

###############################################
# Poisson step
@nb.njit(cache=True)
def poisson(us, vs, p, nx, ny, dx, dy, dt):

    tol = 1.0e-2
    err = 1.0e10
    itp = 0
    ovf = False
    pn  = np.zeros((nx+2,ny+2))
    while(err > tol):

        pn[:,:] = p[:,:]

        for i in range(1,nx+1):
            for j in range(1,ny+1):

                b = (0.5*(us[i+1,j] - us[i-1,j])/dx +
                     0.5*(vs[i,j+1] - vs[i,j-1])/dy)/dt

                p[i,j] = 0.5*((pn[i+1,j] + pn[i,j-1])*dy*dy +
                              (pn[i,j+1] + pn[i-1,j])*dx*dx -
                              b*dx*dx*dy*dy)/(dx**2+dy**2)

        # Domain left (neumann)
        p[ 0,1:-1] = p[ 1,1:-1]

        # Domain right (neumann)
        p[-1,1:-1] = p[-2,1:-1]

        # Domain top (dirichlet)
        p[1:-1,-1] = 0.0

        # Domain bottom (neumann)
        p[1:-1, 0] = p[1:-1, 1]

        # Compute error
        dp  = np.reshape(p - pn, (-1))
        err = np.dot(dp,dp)

        itp += 1
        if (itp > 10000):
            ovf = True
            break

    return itp, ovf


###############################################
# Corrector step
@nb.njit(cache=True)
def corrector(u, v, us, vs, p, nx, ny, dx, dy, dt):

    u[2:-1,1:-1] = us[2:-1,1:-1] - dt*(p[2:-1,1:-1] - p[1:-2,1:-1])/dx
    v[1:-1,2:-1] = vs[1:-1,2:-1] - dt*(p[1:-1,2:-1] - p[1:-1,1:-2])/dy
