# Generic imports
import time

# Custom imports
from rayleigh import *

#######################################
# Generate initial data
#######################################

plot_freq = 100
control = 0.0
s = rayleigh(init=False, n_sgts=1)
s.reset_fields()

start_time = time.time()

for it in range(s.n_warmup):
    print("# it = "+str(it)+" / "+str(s.n_warmup), end='\r')
    obs, rwd, done, trunc, _ = s.step([control])
    if (it>0) and (it%plot_freq == 0):
        s.render(show=False, dump=False)

end_time = time.time()
print("# Loop time = {:f}".format(end_time - start_time))

s.dump("init_field.dat", "init_act.dat")
