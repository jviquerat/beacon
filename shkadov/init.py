# Generic imports
import time

# Custom imports
from shkadov import *

#######################################
# Generate initial data
#######################################

plot_freq = 2000000
control = 0.0
s = shkadov(init=False)
s.reset_fields()
n_it = 2*s.n_warmup

start_time = time.time()

for it in range(2*s.n_warmup):
    print("# it = "+str(it)+" / "+str(n_it), end='\r')
    obs, rwd, done, trunc, _ = s.step([control]*s.n_jets)
    if (it>0) and (it%plot_freq == 0):
        s.render(show=True)

end_time = time.time()
print("# Loop time = {:f}".format(end_time - start_time))

s.dump("init.dat")
