# Generic imports
import time

# Custom imports
from vortex import *

#######################################
# Run without control
#######################################

l = vortex()
l.reset()

# Set parameters for control-free run
n        = l.n_act
sum_rwd  = 0.0
s_time   = time.time()
plt_freq = 1000    # plotting frequency
show     = False # set to True to show while running

for i in range(n):
    act = np.array([-1.0, 0.0])
    obs, rwd, done, trunc, _ = l.step(act)
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    l.render(show=show)

l.render(show=show)
e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
