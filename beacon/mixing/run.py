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
plt_freq = 20    # plotting frequency
show     = False # set to True to show while running

for i in range(n):
    act = np.random.uniform(-1.0, 1.0, 2).tolist()
    obs, rwd, done, trunc, _ = s.step([1.0, -1.0])
    sum_rwd += rwd
    end="\r"
    if (i==n-1): end="\n"
    print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
    if (i%plt_freq == 0): s.render(show=show)

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
print('# Default rwd = '+str(sum_rwd))
