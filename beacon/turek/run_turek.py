# Generic imports
import time

# Custom imports
from turek import *

#######################################
# Run without control
#######################################

# Initialize
#t = turek(cfl=0.1, re=10.0, t_max=20.0)
t = turek(l=10.0, dx=0.2, dy=0.2, t_max=100.0, cfl=0.95, re=100.0)
t.reset_fields()

# Set parameters for control-free run
# n        = s.n_warmup + s.n_act
# sum_rwd  = 0.0
s_time   = time.time()
# plt_freq = 50   # plotting frequency
# show     = True # set to True to show while running

#for i in range(t.n_dt):
while(t.t < t.t_max):
    t.step()
    end = "\r"
    #if (i==t.n_dt-1): end="\n"
    #print("# t= "+str(t.t)+", dt="+str(t.dt)+"                         ", end=end)#+" over "+str(t.n_dt),end=end)

    print('# t= '+'{:f}'.format(t.t)+', dt= '+'{:f}'.format(t.dt), end=end)
t.plot_fields()
t.plot_iterations()

e_time = time.time()
print("# Loop time = {:f}".format(e_time - s_time))
# print('# Default rwd = '+str(sum_rwd))


# for i in range(n):
#     obs, rwd, done, trunc, _ = s.step()
#     sum_rwd += rwd
#     end="\r"
#     if (i==n-1): end="\n"
#     print("# Iteration "+str(i)+", rwd = "+str(rwd)+"          ",end=end)
#     if (i%plt_freq == 0): s.render(show=show)





#######################################################
#t = turek(t_max=40.0, dx=0.1, dy=0.1, cfl=0.1)

