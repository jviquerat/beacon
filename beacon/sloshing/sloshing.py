import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def secondsToStr(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def find_nearest(array, value):
    idx = (np.np.abs(array - value)).argmin()
    return idx


def plot1D(x, h, v=None):

    # Figure setup
    fig, ax = plt.subplots()

    ax.set_xlabel(r'$L [m]$')
    ax.set_ylabel(r'$\eta [m]$', color='tab:red')
    ax.set_xlim([np.min(x), np.max(x)])
    ax.set_ylim([np.min(h), np.max(h)])
    ax.tick_params(axis='y', labelcolor='tab:red')
    ax.plot(x, h, color='tab:red')

    if not v is None:
        ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'tab:blue'
        # we already handled the x-label with ax1
        ax2.set_ylabel('sin', color=color)
        ax2.plot(np.linspace(np.min(x), np.max(x), len(x)+1), v, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.show()

    return


def clean_up_artists(axis, artist_list):
    """
    Try to remove the artists stored in the artist list belonging to the 'axis'.
     axis: clean artists belonging to these axis
     artist_list: list of artist to remove
    return: nothing
    """
    for artist in artist_list:
        try:
            # fist attempt: try to remove collection of contours for instance
            while artist.collections:
                for col in artist.collections:
                    artist.collections.remove(col)
                    try:
                        axis.collections.remove(col)
                    except ValueError:
                        pass

                artist.collections = []
                axis.collections = []
        except AttributeError:
            pass

        # Second attempt, try to remove the text or the line plot
        try:
            artist.remove()
        except (AttributeError, ValueError):
            pass


class TimeSeries1D():

    def __init__(self, x, h, time):
        '''
        Load data for animation
             map2D:     Map used for the simulation
             time:      List of times corresponding to the frames in data.
            return: nothing
        '''

        self.x = x
        self.time = time
        self.h = h

    def createAnimation(self, start_frame=None, end_frame=None, skip_frames=None):
        '''
        Create animation with the data given at init
            start_frame: Start from frame number e.g. 0
            end_frame: Start from frame number e.g. len(labels[:,0,0])
            skip_frames: Amount of frames -1 to skip between every displayed image e.g. 1 (no skipping)
            return: nothing
        '''

        # Set frame control
        if start_frame is None:
            start_frame = 0

        if end_frame is None:
            end_frame = len(self.time)

        if skip_frames is None:
            skip_frames = 1  # 1 - no skipping

        frames = range(start_frame, end_frame, skip_frames)

        # Figure setup
        fig, ax = plt.subplots()

        ax.set_xlabel(r'$L [m]$')
        ax.set_ylabel(r'$\eta [m]$', color='tab:red')
        ax.set_xlim([np.min(self.x), np.max(self.x)])
        ax.set_ylim([0, np.max(np.array(self.h))])
        ax.tick_params(axis='y', labelcolor='tab:red')
        ax.set_title("{}".format("1D Simulation"),
                     transform=ax.transAxes, fontdict=dict(color="blue", size=12))

        changed_artists = list()
        line, = ax.plot(self.x, self.h[0], color='tab:red')
        changed_artists.append(line)

        time_text = ax.text(0.6, 1.05, "{}".format("Time : "+secondsToStr(self.time[0])),
                            transform=ax.transAxes, fontdict=dict(color="black", size=14))
        changed_artists.append(time_text)

        print("\nProcessing animation...")

        def update_plot1D(frame_index):
            """
            Update the line plot of the time step 'frame_index'
            """

            h = self.h[frame_index]

            # Create the contour plot
            line.set_data(self.x, h)

            time_text.set_text("{}".format(
                "Time : "+secondsToStr(self.time[frame_index])))

            return line, time_text

        # Call the animation function. The fargs argument equals the parameter list of update_plot,
        # except the 'frame_index' parameter.
        self.ani = animation.FuncAnimation(
            fig, update_plot1D, frames=frames, interval=1, blit=False, repeat=False)
        # plt.show()

    def saveAnimation(self, fps=60, name='toLazytoName1D'):
        '''
        Save animation after computing it with createAnimation
             fps: Amount of frames displayed each second in the video e.g. 10.
             name: Name of the video. Is stored depending on the call path of the object.
            return: nothing
        '''
        if not hasattr(self, 'ani'):
            print(
                "No animation available. Please call createAnimation on the object before saving it.")
        else:
            print("Saving animation...")
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(
                artist='Me'))  # , bitrate=-1)
            self.ani.save(name+'.mp4', writer=writer)

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################

# Macros "aka implementation cancera"
G = 9.81
L = 1
N = 100
T = 4
DX = L/(N+1)
DT = 1e-3
MU = 0.0
N_PRNT = 10

BDF = 2
DT_TERM = 3/(2*DT) if BDF == 2 else 1/DT

#################################################################
#################################################################
#################################################################
#################################################################
#################################################################
#################################################################


def solveConservative(a_d, v_d):
    h             = np.ones(N)
    q             = np.zeros(N)
    hm            = np.ones(N)
    qm            = np.zeros(N)
    #hv            = np.zeros(N)
    qgh           = np.zeros(N)
    qghm          = np.zeros(N)
    #flux_mass     = np.zeros(N)
    #flux_momentum = np.zeros(N)

    result = [h]
    time   = [0.0]

    def d1lax(u, du, um, nx, dt, dx):

        um[0:nx-1] = 0.5*(u[0:nx-1] + u[1:nx]) - (0.5*dt/dx)*(du[1:nx] - du[0:nx-1])

    def d1o1(u, du, nx, dt, dx):

        u[1:nx-1] = u[1:nx-1] - (dt/dx)*(du[1:nx-1] - du[0:nx-2])

    # def d1tvd(u, du, nx, dx):
    #     phi = np.zeros((nx))
    #     r   = np.zeros((nx))

    #     r[1:nx-1]   = (u[1:nx-1] - u[0:nx-2])/(u[2:nx] - u[1:nx-1] + 1.0e-8)
    #     phi[1:nx-1] = np.maximum(0.0, np.minimum(r[1:nx-1], 1.0))  # mindmod

    #     du[1:nx-1]  = u[1:nx-1] #+ 0.5*phi[1:nx-1]*(u[2:nx]   - u[1:nx-1])
    #     du[1:nx-1] -= u[0:nx-2] #+ 0.5*phi[0:nx-2]*(u[1:nx-1] - u[0:nx-2])
    #     du[1:nx-1] /= dx

    # To reach the desired time
    t = 0.0
    it = 0
    while T > t:

        h[0]  = h[1]
        h[-1] = h[-2]
        #q[0]  = h[0]*v_d(t)
        #q[-1] = h[-1]*v_d(t)

        q[0] = 0.0
        q[-1] = 0.0

        #q[0]  = v_d(t)/h[0]
        #q[-1] = v_d(t)/h[-1]

        qgh[:] = q[:]*q[:]/h[:] + 0.5*G*h[:]*h[:]

        d1lax(h, q,   hm, N, DT, DX)
        d1lax(q, qgh, qm, N, DT, DX)
        qm[:] -= 0.5*DT*a_d(t)

        qghm[:] = qm[:]*qm[:]/hm[:] + 0.5*G*hm[:]*hm[:]

        d1o1(h, qm,   N, DT, DX)
        d1o1(q, qghm, N, DT, DX)
        q[:] -= DT*a_d(t)

        #d1tvd(h*(v-v_d(t)), flux_mass, DX)
        #d1tvd(h*(v-v_d(t))**2 + G/2*h**2, flux_momentum, DX)

        # hv[:]   = h[:]*v[:]
        # hgv2[:] = h[:]*v[:]*v[:] + 0.5*G*h[:]*h[:]

        # d1tvd(hv, flux_mass, N, DX)
        # d1tvd(hgv2, flux_momentum, N, DX)

        # #hv[:]      = h[:]*v[:]
        # h[1:N-1]  -= DT*flux_mass[1:N-1]
        # hv[1:N-1] -= DT*flux_momentum[1:N-1]
        # v[1:N-1]   = hv[1:N-1] / h[1:N-1]

        t += DT

        if (int(1000*(t+DT)/T) - int(1000*t/T)) == 1:
            printProgressBar(int(1000*(t+DT)/T), 1000, prefix='Progress:',
                             suffix='dt = '+str(DT), length=50)
        result.append(np.copy(h))
        time.append(t)

        if (it == 0):
            plt.figure(figsize=(5,2.5))
            x = np.linspace(0, L, N)

        if (it%N_PRNT == 0):
            ax = plt.gca()
            fig = plt.gcf()
            ax.set_xlim([0.0,1.0])
            ax.set_ylim([0.0,2.0])
            ax.plot(x, h)
            plt.pause(0.1)
            plt.clf()

        it += 1

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
        #return 2.0*np.sin(np.pi*t)
        if t < 1:
            return np.sin(np.pi*t)
        else:
            return 0.0

    def a_d(t):
        dt = 1.0e-5
        return (v_d(t+dt) - v_d(t))/dt

    x = np.linspace(0, L, N)
    t, h = solveConservative(a_d, v_d)

    anim = TimeSeries1D(x, h, t)
    anim.createAnimation(skip_frames=40)
    anim.saveAnimation(fps=30)


if __name__ == "__main__":
    main()
