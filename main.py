# Generic imports
import numpy             as np
import matplotlib.pyplot as plt

# Custom imports
from utils import *

### Define parameters
L     = 300          # length of domain
nx    = 3000         # nb of discretization points
dx    = float(L/nx)  # spatial step
dt    = 0.0001        # physical timestep
tmax  = 20.0         # maximum time
nt    = int(tmax/dt) # nb of timesteps
alpha = 6.0/5.0      # physical parameter
beta  = 2.0          # physical parameter
dx2   = dx*dx        # square of spatial step
dx3   = dx*dx*dx     # cube of spatial step
pfrq  = 10000        # plotting frequency
sigma = 5.0e-3       # input noise

### Declare arrays
x    = np.linspace(0, nx, num=nx, endpoint=False)*dx
h    = np.zeros((nx)) # current h
q    = np.zeros((nx)) # current q
hp   = np.zeros((nx)) # previous h
qp   = np.zeros((nx)) # previous q
hpp  = np.zeros((nx)) # previous hp
qpp  = np.zeros((nx)) # previous qp
q2h  = np.zeros((nx)) # current q**2/h
dq2h = np.zeros((nx)) # 1st-order derivative of q**2/h
dddh = np.zeros((nx)) # 3rd-order derivative of h
ddh  = np.zeros((nx)) # 2nd-order derivative of h
dq   = np.zeros((nx)) # 1st-order derivative of q

### Initialize plot
plt.ion()

### Initial solution
h[:]   = 1.0
q[:]   = 1.0
hp[:]  = 1.0
qp[:]  = 1.0
hpp[:] = 1.0
qpp[:] = 1.0

### Time loop
print("Starting loop")
for i in range(nt):
    # Print
    end="\r"
    if (i==nt-1): end="\n"
    print("Iteration "+str(i)+"/"+str(nt)+"          ",end=end)

    # Update previous fields
    hpp[:] = hp[:]
    qpp[:] = qp[:]
    hp[:]  = h[:]
    qp[:]  = q[:]

    # Compute q2h and first spatial derivative
    q2h = q*q/h
    d1(q2h, dq2h, 1, nx-1, dx)

    # Compute q first spatial derivative
    d1(q, dq, 1, nx-1, dx)

    # Compute h third spatial derivative
    d3(h, dddh, 2, nx-2, dx)
    d2(h, ddh,  1, nx-1, dx)
    dddh[nx-2] = (ddh[nx-2] - ddh[nx-3])/dx
    dddh       = h*dddh + h - q/(h*h)

    # Updates
    h[1:nx-1] = (4.0*hp[1:nx-1] - hpp[1:nx-1] - 2.0*dt*dq[1:nx-1])/3.0
    q[1:nx-1] = (4.0*qp[1:nx-1] - qpp[1:nx-1] - 2.0*dt*(alpha*dq2h[1:nx-1] - beta*dddh[1:nx-1]))/3.0

    # Boundary conditions
    h[0]  = 1.0 + np.random.uniform(-sigma, sigma, 1)
    q[0]  = 1.0

    h[nx-2] = h[nx-3]
    h[nx-1] = h[nx-2]

    q[nx-2] = q[nx-3]
    q[nx-1] = q[nx-2]

    # Plot
    if (i%pfrq == 0):
        ax = plt.gca()
        ax.set_ylim([0.0,2.0])
        plt.plot(x,h)
        plt.draw()
        plt.pause(0.0001)
        plt.clf()
