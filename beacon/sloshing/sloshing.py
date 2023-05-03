import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

from Visualization import plot1D, printProgressBar, TimeSeries1D

# Macros
G = 9.81
L = 1
N = 40
T = 4
N_T = 100
DX = L/(N+1)
DT = 5e-4
MU = 0.0

BDF = 2
DT_TERM = 3/(2*DT) if BDF == 2 else 1/DT


DEBUG = True


def solveConservative(v_d):
    h = np.ones(N)
    v = np.zeros(N)
    hv = np.ones(N)
    flux_mass = np.zeros(N-2)
    flux_momentum = np.zeros(N-2)

    result = [h]
    time = [0.0]

    def d1tvd(u, du, dx):
        nx = len(u) - 2
        phi = np.zeros((nx))
        r = np.zeros((nx))
        r[1:nx-1] = (u[1:nx-1] - u[0:nx-2])/(u[2:nx] - u[1:nx-1] + 1.0e-8)
        phi[1:nx-1] = np.maximum(0.0, np.minimum(r[1:nx-1], 1.0))  # mindmod
        # phi[1:nx-1] = (np.abs(r[1:nx-1]) + r[1:nx-1]) / (1 + np.abs(r[1:nx-1]))  # vanLeer

        du[1:nx-1] = u[1:nx-1] + 0.5*phi[1:nx-1]*(u[2:nx] - u[1:nx-1])
        du[1:nx-1] -= u[0:nx-2] + 0.5*phi[0:nx-2]*(u[1:nx-1] - u[0:nx-2])
        du[1:nx-1] /= dx
        return

    # To reach the desired time
    t = 0.0
    while T > t:

        v[0] = v_d(t)
        v[-1] = v_d(t)
        h[0] = h[1]
        h[-1] = h[-2]

        d1tvd(h*(v-v_d(t)), flux_mass, DX)
        d1tvd(h*(v-v_d(t))**2 + G/2*h**2, flux_momentum, DX)

        hv = h*v
        h[1:-1] -= DT*flux_mass
        hv[1:-1] -= DT*flux_momentum
        v[1:-1] /= h[1:-1]

        t += DT

        if (int(1000*(t+DT)/T) - int(1000*t/T)) == 1:
            printProgressBar(int(1000*(t+DT)/T), 1000, prefix='Progress:',
                             suffix='dt = '+str(DT), length=50)
        result.append(np.copy(h))
        time.append(t)

    return np.array(time), np.array(result)


def solveImpNonLinearLongWave(v_d):

    def compute_vrow(v):
        if v > 0:
            return [-v/DX-MU/DX**2, -G/DX, DT_TERM + v/DX + 2*MU/DX**2, G/DX, -MU/DX**2]
        else:
            return [-MU/DX**2, -G/DX, DT_TERM - v/DX + 2*MU/DX**2, G/DX, v/DX - MU/DX**2]

    def compute_hrow(v1, v2):
        return [-v1/(2*DX), 0, DT_TERM + v2/(2*DX)-v1/(2*DX), 0, v2/(2*DX)]

    def assemble_matrix(x, v_d):
        A = np.zeros((2*N+1, 2*N+1))
        A[0, 0] = 1
        A[-1, -1] = 1

        A[1, 1:4] = [DT_TERM + (x[2]-v_d)/(2*DX) -
                     (x[0]-v_d)/DX, 0, (x[2]-v_d)/(2*DX)]
        A[-2, -4:-1] = [-(x[-3]-v_d)/(2*DX), 0, DT_TERM +
                        (x[-1]-v_d)/DX - (x[-3]-v_d)/(2*DX)]

        for i in range(2, 2*N-1, 2):
            A[i, i-2:i+3] = compute_vrow(x[i]-v_d)

        for i in range(3, 2*N-2, 2):
            A[i, i-2:i+3] = compute_hrow(x[i-1]-v_d, x[i+1]-v_d)

        return A

    def assemble_rhs(x, x_, bcs):
        if BDF == 2:
            b = (4*x-x_)/(2*DT)
        else:
            b = x/DT

        b[0] = bcs
        b[-1] = bcs
        return b

    x = np.zeros(2*N+1)
    x[1::2] = 1
    x_old = np.copy(x)

    result = [x[1::2]]
    time = [0.0]

    # To reach the desired time
    t = 0.0
    while T > t:
        b = assemble_rhs(x, x_old, v_d(t))

        for k in range(20):
            A = assemble_matrix(x, v_d(t))
            x = np.linalg.solve(A, b)
            res = np.linalg.norm(x-x_old)
            x_old = np.copy(x)
            # print(f"Residual at iteration {k}: {res}")

        # print("\n\n")

        t += DT

        if (int(1000*(t+DT)/T) - int(1000*t/T)) == 1:
            printProgressBar(int(1000*(t+DT)/T), 1000, prefix='Progress:',
                             suffix='dt = '+str(DT), length=50)
        result.append(np.copy(x[1::2]))
        time.append(t)

    return np.array(time), np.array(result)


def solveNonLinearLongWave(v_d):
    '''
        Solve the linear long wave equation and plot it as an animation
        once all frames are calculated
    '''

    h = np.ones(N)
    v = np.zeros(N+1)

    result = [h]
    time = [0.0]

    # Derivative of eta for velocity equation
    Deta = -G/DX*sparse.diags([-1, 1], [0, 1], shape=(N-1, N)).tocsr()

    # Derivative of velocity*distance in eta equation
    Dvh = -1/DX*sparse.diags([-1, 1], [0, 1], shape=(N, N+1)).tocsr()

    # Upwind derivative for positive velocities
    Dv_forw = -1/DX*sparse.diags([-1, 1], [0, 1], shape=(N-1, N+1)).tocsr()

    # Upwind derivative for negative velocities
    Dv_back = -1/DX*sparse.diags([-1, 1], [1, 2], shape=(N-1, N+1)).tocsr()

    # Viscous dissipation
    Dmu = -DX**-2 * sparse.diags([-1, 2, -1],
                                 [0, 1, 2], shape=(N-1, N+1)).tocsr()

    # Interpolation matrices
    Ih = 0.5*sparse.diags([1, 1], [-1, 0], shape=(N+1, N)).tolil()
    Ih[0, 0] = 1
    Ih[-1, -2] = 0
    Ih = Ih.tocsr()

    # To reach the desired time
    t = 0.0
    while T > t:
        dt = 0.05*DX/max(1e-1, np.max(np.abs(v) + np.sqrt(2*G*Ih.dot(h))))

        v[0] = v_d(t)
        v[-1] = v_d(t)

        conv_v = v[1:-1]-v_d(t)

        v[1:-1] += dt*Deta.dot(h) +\
            dt*(conv_v*(conv_v > 0.0) * Dv_forw.dot(v) +
                conv_v*(conv_v <= 0.0) * Dv_back.dot(v) +
                MU*Dmu.dot(v))

        h += dt*Dvh.dot(Ih.dot(h)*(v-v_d(t)))

        t += dt

        if (int(1000*(t+dt)/T) - int(1000*t/T)) == 1:
            printProgressBar(int(1000*(t+dt)/T), 1000, prefix='Progress:',
                             suffix='dt = '+str(dt), length=50)
        result.append(np.copy(h))
        time.append(t)

    return np.array(time), np.array(result)


def main():

    def v_d(t):
        if t < 1:
            return np.sin(np.pi*t)
        else:
            return 0

    x = np.linspace(0, L, N)
    t, h = solveConservative(v_d)

    anim = TimeSeries1D(x, h, t)
    anim.createAnimation(skip_frames=40)
    anim.saveAnimation(fps=30)


if __name__ == "__main__":
    main()
