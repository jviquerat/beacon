# Generic imports
import time

# Custom imports
from mixing import *

#######################################
# Run without control
#######################################

# Initialize
s = mixing()
s.reset()

# Set parameters for control-free run
n        = s.n_act
sum_rwd  = 0.0
s_time   = time.time()
plt_freq = 1    # plotting frequency
show     = False # set to True to show while running

s.render(show=show)

for i in range(n):
    act = np.random.randint(0, 4)
    obs, rwd, done, trunc, _ = s.step(act)
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    if (i%plt_freq == 0): s.render(show=show)

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
