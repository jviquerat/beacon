# Generic imports
import time

# Custom imports
from shkadov import *

#######################################
# Run without control
#######################################

# Initialize
s = shkadov()
s.reset_fields()

# Set parameters for control-free run
n        = s.n_warmup + s.n_act
sum_rwd  = 0.0
s_time   = time.time()
plt_freq = 50   # plotting frequency
show     = True # set to True to show while running

for i in range(n):
    obs, rwd, done, trunc, _ = s.step()
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    if (i%plt_freq == 0): s.render(show=show)

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
