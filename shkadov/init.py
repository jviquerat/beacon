# Generic imports
import time

# Custom imports
from shkadov import *

#######################################
# Generate initial data
#######################################

s = shkadov()
s.reset_fields()

for i in range(s.n_warmup):
    obs, rwd, done, trunc, _ = s.step()

s.dump("init.dat")
