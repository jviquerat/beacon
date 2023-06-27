# Generic imports
import time
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

###############################################
### Turek class
class turek():

    ### Initialize
    def __init__(self, l=22.0, h=4.0, dx=0.1, dy=0.1, t_max=1.0, cfl=0.95, re=10.0):

        # Set parameters
        self.l     = l
        self.h     = h
        self.dx    = dx
        self.dy    = dy
        self.t_max = t_max
        self.cfl   = cfl
        self.nu    = 0.01
        self.re    = re
        self.umax  = self.re*self.nu/self.h

        # Check sizes compatibility
        self.eps = 1.0e-8
        self.check_size(l, dx, "l")
        self.check_size(h, dy, "h")

        # Compute nb of unknowns
        self.nx = round(self.l/self.dx)
        self.ny = round(self.h/self.dy)

        # Compute obstacle position
        self.x0 = 2.0
        self.y0 = 2.0
        self.r0 = 0.5
        self.c_xmin = round((self.x0-self.r0)/self.dx)
        self.c_xmax = round((self.x0+self.r0)/self.dx)
        self.c_ymin = round((self.y0-self.r0)/self.dy)
        self.c_ymax = round((self.y0+self.r0)/self.dy)

        # Reset fields
        self.reset_fields()

        # Compute timestep
        self.tau = self.l/self.umax
        mdxy     = min(self.dx, self.dy)
        self.dt  = self.cfl*min(self.tau/self.re,
                                self.tau*self.re*mdxy**2/(4.0*self.l**2))

    ### Reset fields
    def reset_fields(self):

        # Set fields
        # Accound for boundary cells
        self.u   = np.zeros((self.nx+2, self.ny+2))
        self.v   = np.zeros((self.nx+2, self.ny+2))
        self.p   = np.zeros((self.nx+2, self.ny+2))
        self.us  = np.zeros((self.nx+2, self.ny+2))
        self.vs  = np.zeros((self.nx+2, self.ny+2))
        self.phi = np.zeros((self.nx+2, self.ny+2))

        # Array to store iterations of poisson resolution
        self.n_itp = np.array([], dtype=np.int16)

        # Set time
        self.it = 0
        self.t  = 0.0

    ### Check size compatibility
    def check_size(self, x, dx, name):

        if (abs(x-round(x/dx)*dx) > self.eps):
            print("Incompatible size: "+name+" must be a multiple of dx")
            exit()

    ### Set boundary conditions
    def set_bc(self):

        # Left wall
        for j in range(1,self.ny+1):
            y           = (j-0.5)*self.dy
            u_pois      = 4.0*self.umax*(self.h-y)*y/(self.h**2)
            self.u[1,j] = u_pois
        self.v[0,2:-1] =-self.v[1,2:-1]

        # Right wall
        self.u[-1,1:-1] = self.u[-2,1:-1]
        self.v[-1,2:-1] =-self.v[-2,2:-1]

        # Top wall
        self.u[1:,-1]   =-self.u[1:,-2]
        self.v[1:-1,-1] = 0.0

        # Bottom wall
        self.u[1:,0]    =-self.u[1:,1]
        self.v[1:-1,1]  = 0.0

    ### Compute starred fields
    def predictor(self):

        predictor(self.u, self.v, self.us, self.vs, self.p,
                  self.nx, self.ny, self.dt, self.dx, self.dy, self.re)

    ### Compute pressure
    def poisson(self):

        self.phi[:,:] = self.p[:,:]

        itp, ovf = poisson(self.us, self.vs, self.u, self.phi, self.nx, self.ny,
                           self.dx, self.dy, self.dt)

        self.p[:,:] += self.phi[:,:]
        #self.p[:,:] = self.phi[:,:]

        self.n_itp = np.append(self.n_itp, np.array([self.it, itp]))

        if (ovf):
            print("\n")
            print("Exceeded max number of iterations in solver")
            self.plot_fields()
            self.plot_iterations()
            exit(1)

    ### Compute updated fields
    def corrector(self):

        corrector(self.u, self.v, self.us, self.vs, self.phi,
                  self.nx, self.ny, self.dx, self.dy, self.dt)

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

        print("# t = "+str(self.t)+" / "+str(self.t_max), end="\r")

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
        p  = np.rot90(p)

        # Plot velocity
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(vn,
                   cmap = 'RdBu_r',
                   vmin = 0.0,
                   vmax = self.umax)

        filename = "velocity.png"
        plt.axis('off')
        plt.savefig(filename, dpi=100)
        plt.close()

        # Plot pressure
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        im = plt.imshow(p,
                   cmap = 'RdBu_r')
                   #vmin =-1.0,
                   #vmax = 1.0)
        plt.colorbar(im)

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

            #conv = 0.5*(
            #    (uE*(u[i+1,j] + u[i,j])     - abs(uE)*(u[i+1,j] - u[i,j])    )/dx -
            #    (uW*(u[i,j]   + u[i-1,j])   - abs(uW)*(u[i,j]   - u[i-1,j])  )/dx +
            #    (uN*(v[i,j+1] + v[i-1,j+1]) - abs(uN)*(v[i,j+1] - v[i-1,j+1]))/dy -
            #    (uS*(v[i,j]   + v[i-1,j])   - abs(uS)*(v[i,j]   - v[i-1,j])  )/dy)

            diff = ((u[i+1,j] - 2.0*u[i,j] + u[i-1,j])/(dx**2) +
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

            #conv = 0.5*(
            #    (uE*(v[i+1,j] + v[i,j])     - abs(uE)*(v[i+1,j] - v[i,j])    )/dx -
            #    (uW*(v[i,j]   + v[i-1,j])   - abs(uW)*(v[i,j]   - v[i-1,j])  )/dx +
            #    (vN*(v[i,j+1] + v[i,j])     - abs(vN)*(v[i,j+1] - v[i,j])    )/dy -
            #    (vS*(v[i,j]   + v[i,j-1])   - abs(vS)*(v[i,j]   - v[i,j-1])  )/dy)

            diff = ((v[i+1,j] - 2.0*v[i,j] + v[i-1,j])/(dx**2) +
                    (v[i,j+1] - 2.0*v[i,j] + v[i,j-1])/(dy**2))/re

            pres = (p[i,j] - p[i,j-1])/dy

            vs[i,j] = v[i,j] + dt*(diff - conv - pres)

###############################################
# Poisson step
@nb.njit(cache=False)
def poisson(us, vs, u, phi, nx, ny, dx, dy, dt):

    tol      = 1.0e-3
    err      = 1.0e10
    itp      = 0
    itmax    = 100000
    ovf      = False
    phi[:,:] = 0.0
    phin     = np.zeros((nx+2,ny+2))
    while(err > tol):

        phin[:,:] = phi[:,:]

        for i in range(2,nx):
            for j in range(1,ny+1):

                b = ((us[i+1,j] - us[i,j])/dx +
                     (vs[i,j+1] - vs[i,j])/dy)/dt

                phi[i,j] = 0.5*((phin[i+1,j] + phin[i-1,j])*dy*dy +
                                (phin[i,j+1] + phin[i,j-1])*dx*dx -
                                b*dx*dx*dy*dy)/(dx*dx+dy*dy)


        # Domain left (dirichlet)
        #phi[ 1,1:-1] = phi[2,1:-1]
        #phi[0,1:-1] = 0.0
        phi[1,1:-1] = (1.0/(dy*dy+2.0*dx*dx))*(dy*dy*phi[2,1:-1] + dx*dx*(phin[1,2:] + phin[1,:-2]) -
                                               (dx*dx*dy*dy/dt)*((us[2,1:-1] - u[1,1:-1])/dx + (vs[1,2:] - vs[1,1:-1])/dy))

        # Domain right (dirichlet)
        phi[-1,1:-1] =-phi[-2,1:-1]
        #phi[-1,1:-1] = 0.0

        # Domain top (neumann)
        phi[1:-1,-1] = phi[1:-1,-2]
        #phi[:,-1] = phi[:,-2]

        # Domain bottom (neumann)
        phi[1:-1, 0] = phi[1:-1, 1]
        #phi[:, 0] = phi[:, 1]

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
@nb.njit(cache=True)
def corrector(u, v, us, vs, phi, nx, ny, dx, dy, dt):

    u[2:-1,1:-1] = us[2:-1,1:-1] - dt*(phi[2:-1,1:-1] - phi[1:-2,1:-1])/dx
    v[1:-1,2:-1] = vs[1:-1,2:-1] - dt*(phi[1:-1,2:-1] - phi[1:-1,1:-2])/dy


