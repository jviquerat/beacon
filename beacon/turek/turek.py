# Generic imports
import time
import numpy             as np
import matplotlib.pyplot as plt
import numba             as nb

###############################################
### Turek class
class turek():

    ### Initialize
    def __init__(self, l=22.0, h=4.0, dx=0.1, dy=0.1, t_max=40.0, cfl=0.1, re=100.0):

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
        self.re    = re

        # Check sizes compatibility
        self.eps = 1.0e-8
        self.check_size(l, dx, "l")
        self.check_size(h, dy, "h")

        # Compute nb of unknowns
        self.nx = round(self.l/self.dx)
        self.ny = round(self.h/self.dy)
        self.nc = self.nx*self.ny

        # Compute timestep
        self.dt   = self.cfl*min(self.dx,self.dy)
        self.n_dt = round(self.t_max/self.dt)

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

    ### Reset fields
    def reset_fields(self):

        # Set fields
        # Accound for boundary cells
        self.u  = np.zeros((self.nx+2, self.ny+2))
        self.v  = np.zeros((self.nx+2, self.ny+2))
        self.p  = np.zeros((self.nx+2, self.ny+2))

        self.us = np.zeros((self.nx+2, self.ny+2))
        self.vs = np.zeros((self.nx+2, self.ny+2))

        # Array to store iterations of poisson resolution
        self.n_itp = np.zeros((self.n_dt,2), dtype=np.int16)

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

        # Set zero in obstacle
        self.u[self.c_xmin:self.c_xmax+1,self.c_ymin:self.c_ymax+1] = 0.0
        self.v[self.c_xmin:self.c_xmax+1,self.c_ymin:self.c_ymax+1] = 0.0
        self.p[self.c_xmin:self.c_xmax+1,self.c_ymin:self.c_ymax+1] = 0.0

        # No-slip BC on obstacle bottom
        self.u[self.c_xmin:self.c_xmax,self.c_ymin+1] =-self.u[self.c_xmin:self.c_xmax,self.c_ymin]
        self.v[self.c_xmin:self.c_xmax,self.c_ymin+1] = 0.0

        # No-slip BC on obstacle top
        self.u[self.c_xmin:self.c_xmax,self.c_ymax] =-self.u[self.c_xmin:self.c_xmax,self.c_ymax]
        self.v[self.c_xmin:self.c_xmax,self.c_ymax] = 0.0

        # No-slip BC on obstacle left
        self.u[self.c_xmin+1,self.c_ymin:self.c_ymax] = 0.0
        self.v[self.c_xmin+1,self.c_ymin:self.c_ymax] =-self.v[self.c_xmin,self.c_ymin:self.c_ymax]

        # No-slip BC on obstacle right
        self.u[self.c_xmax,self.c_ymin:self.c_ymax] = 0.0
        self.v[self.c_xmax,self.c_ymin:self.c_ymax] =-self.v[self.c_xmax,self.c_ymin:self.c_ymax]

    ### Compute starred fields
    def predictor(self):

        predictor(self.u,   self.v,   self.us, self.vs,
                  self.idx, self.idy, self.nx, self.ny, self.dt, self.re)

    ### Compute pressure
    def poisson(self):

        n_itp = poisson(self.us, self.vs, self.p,
                        self.dx, self.dy, self.idx, self.idy,
                        self.nx, self.ny, self.dt,  self.ifdxy)
        self.n_itp[self.it,0] = self.it
        self.n_itp[self.it,1] = n_itp

    ### Compute updated fields
    def corrector(self):

        corrector(self.u,   self.v,   self.us, self.vs, self.p,
                  self.idx, self.idy, self.nx, self.ny, self.dt)

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
    def plot(self):

        nx = self.nx
        ny = self.ny

        # Recreate fields at cells centers
        u = np.zeros((self.nx, self.ny))
        v = np.zeros((self.nx, self.ny))

        u[0:nx,0:ny] = 0.5*(self.u[0:nx,1:ny+1] + self.u[1:nx+1,1:ny+1])
        v[0:nx,0:ny] = 0.5*(self.v[1:nx+1,0:ny] + self.u[1:nx+1,1:ny+1])

        # Compute norm
        vn = np.sqrt(u**2+v**2)

        # Mask obstacles
        vn[self.c_xmin:self.c_xmax,self.c_ymin:self.c_ymax] = -1.0
        vn = np.ma.masked_where((vn < 0.0), vn)
        vn = np.rot90(vn)

        # Plot
        plt.clf()
        fig, ax = plt.subplots(figsize=plt.figaspect(vn))
        fig.subplots_adjust(0,0,1,1)
        plt.imshow(vn,
                   cmap = 'RdBu_r',
                   vmin = 0.0,
                   vmax = 1.5)

        filename = "field.png"
        plt.axis('off')
        plt.savefig(filename, dpi=200)
        plt.close()

        plt.clf()
        fig, ax = plt.subplots(1,1,figsize=(5,5))
        ax.plot(self.n_itp[:,0], self.n_itp[:,1], color='blue')
        ax.grid(True)
        fig.tight_layout()
        filename = "iterations.png"
        fig.savefig(filename)
        np.savetxt("iterations.dat", self.n_itp, fmt='%d')

###############################################
# Predictor step
@nb.njit(cache=True)
def predictor(u, v, us, vs, idx, idy, nx, ny, dt, re):

    d2ux = np.zeros((nx+2,ny+2))
    d2uy = np.zeros((nx+2,ny+2))
    udux = np.zeros((nx+2,ny+2))
    vduy = np.zeros((nx+2,ny+2))
    vloc = np.zeros((nx+2,ny+2))

    d2ux[1:nx+1,1:ny+1] = (u[0:nx,1:ny+1] - 2.0*u[1:nx+1,1:ny+1] + u[2:nx+2,1:ny+1])*idx*idx
    d2uy[1:nx+1,1:ny+1] = (u[1:nx+1,0:ny] - 2.0*u[1:nx+1,1:ny+1] + u[1:nx+1,2:ny+2])*idy*idy

    vloc[1:nx+1,1:ny+1] = 0.25*(v[0:nx,1:ny+1] + v[1:nx+1,1:ny+1] + v[0:nx,2:ny+2] + v[1:nx+1,2:ny+2])
    udux[1:nx+1,1:ny+1] = u[1:nx+1,1:ny+1]*(u[2:nx+2,1:ny+1] - u[0:nx,1:ny+1])*0.5*idx
    vduy[1:nx+1,1:ny+1] = vloc[1:nx+1,1:ny+1]*(u[1:nx+1,2:ny+2] - u[1:nx+1,0:ny])*0.5*idy

    us[:,:] = u[:,:] + dt*((d2ux[:,:] + d2uy[:,:])/re - (udux[:,:] + vduy[:,:]))

    d2vx = np.zeros((nx+2,ny+2))
    d2vy = np.zeros((nx+2,ny+2))
    udvx = np.zeros((nx+2,ny+2))
    vdvy = np.zeros((nx+2,ny+2))
    uloc = np.zeros((nx+2,ny+2))

    d2vx[1:nx+1,1:ny+1] = (v[0:nx,1:ny+1] - 2.0*v[1:nx+1,1:ny+1] + v[2:nx+2,1:ny+1])*idx*idx
    d2vy[1:nx+1,1:ny+1] = (v[1:nx+1,0:ny] - 2.0*v[1:nx+1,1:ny+1] + v[1:nx+1,2:ny+2])*idy*idy

    uloc[1:nx+1,1:ny+1] = 0.25*(u[1:nx+1,0:ny] + u[1:nx+1,1:ny+1] + u[2:nx+2,0:ny] + u[2:nx+2,1:ny+1])
    udvx[1:nx+1,1:ny+1] = uloc[1:nx+1,1:ny+1]*(v[2:nx+2,1:ny+1] - v[0:nx,1:ny+1])*0.5*idx
    vdvy[1:nx+1,1:ny+1] = v[1:nx+1,1:ny+1]*(v[1:nx+1,2:ny+2] - v[1:nx+1,0:ny])*0.5*idy

    vs[:,:] = v[:,:] + dt*((d2vx[:,:] + d2vy[:,:])/re - (udvx[:,:] + vdvy[:,:]))

###############################################
# Poisson step
@nb.njit(cache=True)
def poisson(us, vs, p, dx, dy, idx, idy, nx, ny, dt, ifdxy):

    b = np.zeros((nx+2,ny+2))
    b[1:nx+1,1:ny+1] = ((us[2:nx+2,1:ny+1] - us[0:nx,1:ny+1])*0.5*idx +
                        (vs[1:nx+1,2:ny+2] - vs[1:nx+1,0:ny])*0.5*idy)/dt

    tol = 1.0e-2
    err = 1.0e10
    itp = 0
    while(err > tol):
        p_old = p.copy()
        p[1:nx+1,1:ny+1] = ((p_old[2:nx+2,1:ny+1] + p_old[0:nx,1:ny+1])*dy*dy +
                            (p_old[1:nx+1,2:ny+2] + p_old[1:nx+1,0:ny])*dx*dx -
                            b[1:nx+1,1:ny+1]*dx*dx*dy*dy)*ifdxy

        dp  = np.reshape(p - p_old, (-1))
        err = np.dot(dp,dp)

        p[-1,1:ny+1] = p[-2,1:ny+1]
        p[ 0,1:ny+1] = p[ 1,1:ny+1]

        itp += 1
        if (itp > 10000): break

    return itp

###############################################
# Corrector step
@nb.njit(cache=True)
def corrector(u, v, us, vs, p, idx, idy, nx, ny, dt):

    u[1:nx+1,1:ny+1] = us[1:nx+1,1:ny+1] - dt*(p[1:nx+1,1:ny+1] - p[0:nx,1:ny+1])*idx
    v[1:nx+1,1:ny+1] = vs[1:nx+1,1:ny+1] - dt*(p[1:nx+1,1:ny+1] - p[1:nx+1,0:ny])*idy