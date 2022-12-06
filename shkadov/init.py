# Generic imports
import time

# Custom imports
from shkadov import *

#######################################
# Generate initial data
#######################################

plot_freq = 10
s = shkadov(init=False)
s.reset_fields()

for i in range(s.n_warmup):
    obs, rwd, done, trunc, _ = s.step()
    if (i%plot_freq == 0):
        s.render(show=True)

#s.dump("init.dat")
