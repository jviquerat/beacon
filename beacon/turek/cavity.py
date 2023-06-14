# Generic imports
import time
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

###############################################
### Cavity class
class cavity():

    ### Initialize
    def __init__(self, l=1.0, h=1.0, dx=0.01, dy=0.01, t_max=1.0, cfl=0.1, re=100.0):

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

        # Term including starred velocities
        b            = np.zeros((self.nx+2,self.ny+2))
        b[1:-1,1:-1] = ((self.us[2:,1:-1] - self.us[:-2,1:-1])*0.5*self.idx +
                        (self.vs[1:-1,2:] - self.vs[1:-1,:-2])*0.5*self.idy)/self.dt

        tol = 1.0e-2
        err = 1.0e10
        itp = 0
        ovf = False
        pn  = np.zeros((self.nx+2,self.ny+2))
        while(err > tol):

            pn[:,:] = self.p[:,:]
            self.p[1:-1,1:-1] = ((pn[2:,1:-1] + pn[1:-1,:-2])*self.dy*self.dy +
                                 (pn[1:-1,2:] + pn[:-2,1:-1])*self.dx*self.dx -
                                 b[1:-1,1:-1]*self.dx*self.dx*self.dy*self.dy)*self.ifdxy

            # Domain left (neumann)
            self.p[ 0,1:-1] = self.p[ 1,1:-1]

            # Domain right (neumann)
            self.p[-1,1:-1] = self.p[-2,1:-1]

            # Domain top (dirichlet)
            self.p[1:-1,-1] = 0.0

            # Domain bottom (neumann)
            self.p[1:-1, 0] = self.p[1:-1, 1]

            # Compute error
            dp  = np.reshape(self.p - pn, (-1))
            err = np.dot(dp,dp)

            itp += 1
            if (itp > 10000):
                ovf = True
                break

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

        # if (ovf):
        #     print("\n")
        #     print("Exceeded max number of iterations in solver")
        #     self.plot_fields()
        #     self.plot_iterations()
        #     exit(1)

    ### Compute updated fields
    def corrector(self):

        self.u[2:-1,1:-1] = self.us[2:-1,1:-1] - self.dt * (self.p[2:-1,1:-1] - self.p[1:-2,1:-1])/self.dx
        self.v[1:-1,2:-1] = self.vs[1:-1,2:-1] - self.dt * (self.p[1:-1,2:-1] - self.p[1:-1,1:-2])/self.dy

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
#@nb.njit(cache=True)
def predictor(u, v, us, vs, nx, ny, dt, dx, dy, re):

    for i in range(2,nx+1):
        for j in range(1,ny+1):
            uE = 0.5*(u[i+1,j] + u[i,j])
            uW = 0.5*(u[i,j]   + u[i-1,j])

            uN = 0.5*(u[i,j+1] + u[i,j])
            uS = 0.5*(u[i,j] + u[i,j-1])

            vN = 0.5*(v[i,j+1] + v[i-1,j+1])
            vS = 0.5*(v[i,j] + v[i-1,j])

            # convection = - d(uu)/dx - d(vu)/dy
            convection = - (uE*uE - uW*uW)/dx - (uN*vN - uS*vS)/dy

            # diffusion = d2u/dx2 + d2u/dy2
            diffusion = ( (u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/dx/dx + (u[i,j+1] - 2.0*u[i,j] + u[i,j-1])/dy/dy )/re

            us[i,j] = u[i,j] + dt *(convection + diffusion)

    for i in range(1,nx+1):
        for j in range(2,ny+1):
            vE = 0.5*(v[i+1,j] + v[i,j])
            vW = 0.5*(v[i,j] + v[i-1,j])

            uE = 0.5*(u[i+1,j] + u[i+1,j-1])
            uW = 0.5*(u[i,j] + u[i,j-1])

            vN = 0.5*(v[i,j+1] + v[i,j])
            vS = 0.5*(v[i,j] + v[i,j-1])

            # convection = d(uv)/dx + d(vv)/dy
            convection = - (uE*vE - uW*vW)/dx - (vN*vN - vS*vS)/dy

            # diffusion = d2u/dx2 + d2u/dy2
            diffusion = ( (v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/dx/dx + (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/dy/dy )/re

            vs[i,j] = v[i,j] + dt*(convection + diffusion)

###############################################
# Poisson step
#@nb.njit(cache=True)
#def poisson(us, vs, phi, dx, dy, idx, idy, nx, ny, dt, ifdxy,
#c_xmin, c_xmax, c_ymin, c_ymax):

    # # Set zero in obstacle
    # us[c_xmin:c_xmax,c_ymin:c_ymax] = 0.0
    # vs[c_xmin:c_xmax,c_ymin:c_ymax] = 0.0

    # # Term including starred velocities
    # b = np.zeros((nx+2,ny+2))
    # b[1:nx+1,1:ny+1] = ((us[2:nx+2,1:ny+1] - us[0:nx,1:ny+1])*0.5*idx +
    #                     (vs[1:nx+1,2:ny+2] - vs[1:nx+1,0:ny])*0.5*idy)/dt
    # #(vs[1:nx+1,2:ny+2] - vs[1:nx+1,0:ny])*0.5*idy)*3.0/(2.0*dt)


    # #b[c_xmin:c_xmax,c_ymin:c_ymax] = 0.0

    # tol = 1.0e-2
    # err = 1.0e10
    # itp = 0
    # ovf = False
    # phi_old = np.zeros((nx+2,ny+2))
    # while(err > tol):

    #     phi_old[:,:] = phi[:,:]
    #     phi[1:nx+1,1:ny+1] = ((phi_old[2:nx+2,1:ny+1] + phi_old[0:nx,1:ny+1])*dy*dy +
    #                           (phi_old[1:nx+1,2:ny+2] + phi_old[1:nx+1,0:ny])*dx*dx -
    #                           b[1:nx+1,1:ny+1]*dx*dx*dy*dy)*ifdxy

    #     # # Domain left (neumann)
    #     phi[ 0,1:ny+1] = phi[ 1,1:ny+1]

    #     # # Domain right (dirichlet)
    #     phi[-1,1:ny+1] =-phi[-2,1:ny+1]

    #     # # Domain top (neumann)
    #     #phi[1:nx+1,-1] = phi[1:nx+1,-2]

    #     # # Domain bottom (neumann)
    #     #phi[1:nx+1, 0] = phi[1:nx+1, 1]

    #     # # Obstacle left (neumann)
    #     #phi[c_xmin,c_ymin:c_ymax] = phi[c_xmin-1,c_ymin:c_ymax]

    #     # # Obstacle right (neumann)
    #     # phi[c_xmax,c_ymin:c_ymax] = phi[c_xmax+1,c_ymin:c_ymax]

    #     # # Obstacle top (neumann)
    #     # phi[c_xmin:c_xmax,c_ymax] = phi[c_xmin:c_xmax,c_ymax+1]

    #     # # Obstacle bottom (neumann)
    #     # phi[c_xmin:c_xmax,c_ymin] = phi[c_xmin:c_xmax,c_ymin-1]

    #     dp  = np.reshape(phi - phi_old, (-1))
    #     err = np.dot(dp,dp)

    #     itp += 1
    #     if (itp > 10000):
    #         ovf = True
    #         break

    # return ovf, itp

###############################################
# Corrector step
#@nb.njit(cache=True)
#def corrector(u, v, us, vs, phi, idx, idy, nx, ny, dt):

    # u[1:nx+1,1:ny+1] = us[1:nx+1,1:ny+1] - dt*(phi[1:nx+1,1:ny+1] - phi[0:nx,1:ny+1])*idx
    # v[1:nx+1,1:ny+1] = vs[1:nx+1,1:ny+1] - dt*(phi[1:nx+1,1:ny+1] - phi[1:nx+1,0:ny])*idy
