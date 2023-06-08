# Generic imports
import time

# Custom imports
from sloshing import *

#######################################
# Generate initial data
#######################################

plot_freq = 1
control = 0.0
s = sloshing(init=False)
s.reset_fields()

start_time = time.time()
t = 0.0

for it in range(s.n_warmup):
    print("# it = "+str(it)+" / "+str(s.n_warmup), end='\r')
    u = s.signal(t, s.dt_act)
    obs, rwd, done, trunc, _ = s.step([u])
    t += s.dt_act
    if (it>0) and (it%plot_freq == 0):
        s.render(show=True)

end_time = time.time()
print("# Loop time = {:f}".format(end_time - start_time))

s.dump("init_field.dat", "init_control.dat")
