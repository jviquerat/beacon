# Generic imports
import time

# Custom imports
from lorenz import *

#######################################
# Run without control
#######################################

l = lorenz()
l.reset()

# Set parameters for control-free run
n        = l.n_act
sum_rwd  = 0.0
s_time   = time.time()
plt_freq = 10    # plotting frequency
show     = False # set to True to show while running

for i in range(n):
    act = np.random.randint(0, 3)
    obs, rwd, done, trunc, _ = l.step(act)
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    if (i%plt_freq == 0): l.render(show=show)

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
