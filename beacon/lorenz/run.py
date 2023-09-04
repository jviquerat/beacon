# Custom imports
from lorenz import *

l = lorenz(0, '.')
l.reset(0)
#u = np.array([0.0, 0.0, 0.0, 0.0])
u = np.array([-0.34429, 0.25416, 1.04439, 0.17369])
l.cost(u)
l.render(u)
l.render_gif(u)
