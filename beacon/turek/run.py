# Generic imports
import time
import numpy             as np
import matplotlib.pyplot as plt

###############################################
### Turek class
class turek():

    ### Initialize
    def __init__(self, l=22.0, h=4.1, dx=0.1, dy=0.1, t_max=1.0, cfl=0.1):

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

        # Check compatibility
        self.eps = 1.0e-8
        self.check_size(l, dx, "l")
        self.check_size(h, dy, "h")

        # Compute nb of cells
        self.nx  = round(self.l/self.dx)
        self.ny  = round(self.h/self.dy)
        self.nc  = self.nx*self.ny

        # Compute timestep
        self.dt   = self.cfl*min(self.dx,self.dy)
        self.n_dt = round(self.t_max/self.dt)

        # Set fluid properties
        self.rho = 1.0
        self.mu  = 0.01
        self.re  = self.rho*1.0*1.0/self.mu

        # Reset fields
        self.reset()

    ### Reset fields
    def reset(self):

        # Set fields
        # Accound for boundary cells
        self.u = np.zeros((self.nx+2, self.ny+2))
        self.v = np.zeros((self.nx+2, self.ny+2))
        self.p = np.zeros((self.nx+2, self.ny+2))

        # Set time
        self.t = 0.0

    ### Check size compatibility
    def check_size(self, x, dx, name):

        if (abs(x-round(x/dx)*dx) > self.eps):
            print("Incompatible size: "+name+" must be a multiple of dx")
            exit()

    ### Set boundary conditions
    def set_bc(self):

        nx = self.nx
        ny = self.ny

        # Poiseuille at inlet
        for j in range(self.ny):
            y             = j*self.dy
            u_pois        = 4.0*(self.h-y)*y/(self.h**2)
            self.u[0,j+1] = u_pois
        self.v[0,1:ny+1] =-self.v[1,1:ny+1]

        # No-slip BC at top
        self.u[1:nx+1,-1] =-self.u[1:nx+1,-2]
        self.v[1:nx+1,-1] = 0.0

        # No-slip BC at bottom
        self.u[1:nx+1,0] =-self.u[1:nx+1,1]
        self.v[1:nx+1,0] = 0.0
        self.v[1:nx+1,1] = 0.0

        # Output BC at outlet
        self.u[-1,1:ny+1] = self.u[-2,1:ny+1] # Neumann   for u
        self.v[-1,1:ny+1] =-self.v[-2,1:ny+1] # Dirichlet for v
        self.p[-1,1:ny+1] =-self.p[-2,1:ny+1] # Free boundary for pressure

    ### Compute starred fields
    def predictor(self):

        nx = self.nx
        ny = self.ny

        d2ux = np.zeros((self.nx+2,self.ny+2))
        d2uy = np.zeros((self.nx+2,self.ny+2))
        udux = np.zeros((self.nx+2,self.ny+2))
        vduy = np.zeros((self.nx+2,self.ny+2))
        vloc = np.zeros((self.nx+2,self.ny+2))
        us   = np.zeros((self.nx+2,self.ny+2))

        d2ux[1:nx+1,1:ny+1] = (self.u[0:nx,1:ny+1] - 2.0*self.u[1:nx+1,1:ny+1] + self.u[2:nx+2,1:ny+1])*self.idx*self.idx
        d2uy[1:nx+1,1:ny+1] = (self.u[1:nx+1,0:ny] - 2.0*self.u[1:nx+1,1:ny+1] + self.u[1:nx+1,2:ny+2])*self.idy*self.idy

        vloc[1:nx+1,1:ny+1] = 0.25*(self.v[0:nx,1:ny+1] + self.v[1:nx+1,1:ny+1] + self.v[0:nx,2:ny+2] + self.v[1:nx+1,2:ny+2])
        udux[1:nx+1,1:ny+1] = self.u[1:nx+1,1:ny+1]*(self.u[2:nx+2,1:ny+1] - self.u[0:nx,1:ny+1])*0.5*self.idx
        vduy[1:nx+1,1:ny+1] = vloc[1:nx+1,1:ny+1]*(self.u[1:nx+1,2:ny+2] - self.u[1:nx+1,0:ny])*0.5*self.idy

        self.us = self.u + self.dt*((d2ux + d2uy)/self.re - (udux + vduy))

        d2vx = np.zeros((self.nx+2,self.ny+2))
        d2vy = np.zeros((self.nx+2,self.ny+2))
        udvx = np.zeros((self.nx+2,self.ny+2))
        vdvy = np.zeros((self.nx+2,self.ny+2))
        uloc = np.zeros((self.nx+2,self.ny+2))
        vs   = np.zeros((self.nx+2,self.ny+2))

        d2vx[1:nx+1,1:ny+1] = (self.v[0:nx,1:ny+1] - 2.0*self.v[1:nx+1,1:ny+1] + self.v[2:nx+2,1:ny+1])*self.idx*self.idx
        d2vy[1:nx+1,1:ny+1] = (self.v[1:nx+1,0:ny] - 2.0*self.v[1:nx+1,1:ny+1] + self.v[1:nx+1,2:ny+2])*self.idy*self.idy

        uloc[1:nx+1,1:ny+1] = 0.25*(self.u[1:nx+1,0:ny] + self.u[1:nx+1,1:ny+1] + self.u[2:nx+2,0:ny] + self.u[2:nx+2,1:ny+1])
        udvx[1:nx+1,1:ny+1] = uloc[1:nx+1,1:ny+1]*(self.v[2:nx+2,1:ny+1] - self.v[0:nx,1:ny+1])*0.5*self.idx
        vdvy[1:nx+1,1:ny+1] = self.v[1:nx+1,1:ny+1]*(self.v[1:nx+1,2:ny+2] - self.v[1:nx+1,0:ny])*0.5*self.idy

        self.vs = self.v + self.dt*((d2vx + d2vy)/self.re - (udvx + vdvy))

    ### Compute pressure
    def poisson(self):

        nx = self.nx
        ny = self.ny

        b = np.zeros((self.nx+2,self.ny+2))
        b[1:nx+1,1:ny+1] = ((self.u[2:nx+2,1:ny+1] - self.u[1:nx+1,1:ny+1])*self.idx +
                            (self.v[1:nx+1,2:ny+2] - self.v[1:nx+1,1:ny+1])*self.idy)/self.dt


        tol = 1.0e-3
        err = 1.0e10
        itp = 0
        while(err > tol):
            p_old = self.p.copy()
            self.p[1:nx+1,1:ny+1] = ((p_old[2:nx+2,1:ny+1] + p_old[0:nx,1:ny+1])*self.dy*self.dy +
                                     (p_old[1:nx+1,2:ny+2] + p_old[1:nx+1,0:ny])*self.dx*self.dx -
                                     b[1:nx+1,1:ny+1]*self.dx*self.dx*self.dy*self.dy)*self.ifdxy

            err = np.amax(abs(self.p-p_old))

            self.p[-1,1:ny+1]  = self.p[-2,1:ny+1]
            self.p[ 0,1:ny+1]  = self.p[ 1,1:ny+1]
            #self.p[ 1:nx+1, 0] = self.p[ 1:nx+1, 1]
            #self.p[ 1:nx+1,-1] = self.p[ 1:nx+1,-2]

            itp += 1
        print(itp)

    ### Compute updated fields
    def corrector(self):

        nx = self.nx
        ny = self.ny

        dpx = np.zeros((self.nx+2,self.ny+2))
        dpy = np.zeros((self.nx+2,self.ny+2))

        dpx[1:nx+1,1:ny+1] = (self.p[1:nx+1,1:ny+1] - self.p[0:nx,1:ny+1])*self.idx
        dpy[1:nx+1,1:ny+1] = (self.p[1:nx+1,1:ny+1] - self.p[1:nx+1,0:ny])*self.idy

        self.u[1:nx+1,1:ny+1] = self.us[1:nx+1,1:ny+1] - self.dt*dpx[1:nx+1,1:ny+1]
        self.v[1:nx+1,1:ny+1] = self.vs[1:nx+1,1:ny+1] - self.dt*dpy[1:nx+1,1:ny+1]

    ### Take one step
    def step(self):

        self.set_bc()
        self.predictor()
        self.poisson()
        self.corrector()

        self.t += self.dt

    ### Print
    def print(self):

        print('# t = '+str(self.t)+' / '+str(self.t_max), end='\r')

    ### Plot field norm
    def plot(self):

        # Copy without ghots cells
        u = self.u[1:self.nx+1,1:self.nx+1].copy()
        v = self.v[1:self.nx+1,1:self.nx+1].copy()
        p = self.p[1:self.nx+1,1:self.nx+1].copy()

        # u = self.u.copy()
        # v = self.v.copy()
        # p = self.p.copy()

        # Compute norm
        nrm = np.rot90(np.sqrt(u**2+v**2))

        #nrm = np.rot90(p)

        # Mask obstacles
        #v[np.where(lattice.lattice > 0.0)] = -1.0
        #vm = np.ma.masked_where((v < 0.0), v)
        #vm = np.rot90(vm)

        # Plot
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(nrm))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(nrm,
                   cmap = 'RdBu_r',
                   vmin =-1.0,
                   vmax = 1.0,
                   interpolation = 'spline16')

        filename = "test.png"
        plt.axis('off')
        plt.savefig(filename, dpi=200)
        plt.close()

t = turek(t_max=10.0, dx=0.1, dy=0.1, cfl=0.09)
for i in range(t.n_dt):
    t.step()
t.plot()
