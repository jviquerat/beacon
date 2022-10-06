# Generic imports
import numpy as np

# 2nd order first derivative of given field
# u  : field
# du : derivative field
# s  : start index to compute derivative
# e  : end index to compute derivative
def d1(u, du, s, e, dx):

    du[s:e] = (u[s+1:e+1] - u[s-1:e-1])/(2.0*dx)

# 2nd order second derivative of given field
# u  : field
# du : derivative field
# s  : start index to compute derivative
# e  : end index to compute derivative
def d2(u, du, s, e, dx):

    du[s:e] = (u[s+1:e+1] - u[s-1:e-1] + 2.0*u[s:e])/(dx*dx)

# 2nd order third derivative of given field
# u  : field
# du : derivative field
# s  : start index to compute derivative
# e  : end index to compute derivative
def d3(u, du, s, e, dx):

    du[s:e] = (u[s+2:e+2] - 2.0*u[s+1:e+1] + 2.0*u[s-1:e-1] - u[s-2:e-2])/(2.0*dx*dx*dx)
