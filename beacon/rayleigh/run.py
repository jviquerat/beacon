# Generic imports
import time

# Custom imports
from rayleigh import *

#######################################
# Run without control
#######################################

# Initialize
s = rayleigh(init=True)
s.reset()

# Set parameters for control-free run
n        = s.n_act
sum_rwd  = 0.0
s_time   = time.time()
plt_freq = 10    # plotting frequency
show     = False # set to True to show while running

for i in range(n):
    act = np.random.uniform(-1.0, 1.0, 10).tolist()
    obs, rwd, done, trunc, _ = s.step(act)
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    if (i%plt_freq == 0): s.render(show=show)

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
