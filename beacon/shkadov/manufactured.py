# Generic imports
import time
import math as m
import matplotlib.pyplot as plt

# Custom imports
from shkadov import *

#######################################
# Check discretization with manufactured solution
#######################################

plot_freq = 20
control = 0.0
s = shkadov(init=False)
s.reset_fields()
x = np.linspace(0, s.nx, num=s.nx, endpoint=False)*s.dx

omega = 2.3
k     = 0.77
t     = 1.2

# Initialize
for i in range(s.nx):
    s.h[i] = m.cos(omega*t-k*i*s.dx)
    s.q[i] = (omega/k)*m.cos(omega*t-k*i*s.dx)

# Initial rendering
s.render(show=True)

# Check q first derivative
d1tvd(s.q, s.dq, s.nx, s.dx)
dq_ex = np.zeros((s.nx))
for i in range(s.nx):
    dq_ex[i] = omega*m.sin(omega*t-k*i*s.dx)
sum_dq = dq_ex - s.dq

plt.plot(x,dq_ex, label="exact")
plt.plot(x,s.dq, label="computed")
plt.plot(x,sum_dq, label="sum")
plt.legend()
plt.show()

# Check h third derivative
d3o2u(s.h, s.dddh, s.nx, s.dx)
dddh_ex = np.zeros((s.nx))
for i in range(s.nx):
    dddh_ex[i] = -k**3*m.sin(omega*t-k*i*s.dx)
sum_dddh = dddh_ex - s.dddh
plt.plot(x,dddh_ex, label="exact")
plt.plot(x,s.dddh, label="computed")
plt.plot(x,sum_dddh, label="sum")
plt.legend()
plt.show()

# Check q**2/h first derivative
s.q2h[:] = s.q[:]*s.q[:]/(s.h[:] + s.eps)
d1tvd(s.q2h, s.dq2h, s.nx, s.dx)
dq2h_ex = np.zeros((s.nx))
for i in range(s.nx):
    dq2h_ex[i] = ((omega**2)/k)*m.sin(omega*t-k*i*s.dx)
sum_dq2h = dq2h_ex - s.dq2h
plt.plot(x,dq2h_ex, label="exact")
plt.plot(x,s.dq2h, label="computed")
plt.plot(x,sum_dq2h, label="sum")
plt.legend()
plt.show()
