import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.widgets import Slider
from numba import njit, prange, cuda
from matplotlib.animation import FuncAnimation, FFMpegWriter

time_step = 0.005
x = np.linspace(0, 100, 10000)
wmap = np.sin(np.pi * 5 * x / 100)  # Example 1D data
# wmap = 2/(1+np.exp(-(x - 50)/10))-1
c = 3  # Wave speed
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
l, = plt.plot(x, wmap)
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(ax_slider, 'Time', 0, 100, valinit=0)


  # cannot run in parallel because of call to previous element in t_derivatives and w_maps arrays
def iterate(wmap, time_step, num_steps):
    wmaps_arr = np.empty((num_steps, *wmap.shape))
    wmaps_arr[0] = wmap.copy()
    t_derivatives_arr = np.empty_like(wmaps_arr)
    threadsperblock = 32
    blockspergrid = (wmap.size + (threadsperblock - 1)) // threadsperblock
    start_iterations = time.time()
    for i in range(1, num_steps):
        # update t_derivatives
        lap = np.zeros_like(wmaps_arr[i - 1])
        custom_laplacian_gpu[blockspergrid, threadsperblock](wmaps_arr[i - 1], lap)
        t_derivatives_arr[i] = t_derivatives_arr[i - 1] +  lap * c ** 2
        # add t_derivative to wmap
        wmaps_arr[i] = wmaps_arr[i - 1] + time_step * t_derivatives_arr[i]
    print("Iterations took ", time.time() - start_iterations)
    return wmaps_arr


@njit(parallel=True)  # can run in parallel because of no call to previous element in laplacian array
def custom_laplacian(array):
    lap = np.zeros_like(array)
    for i in prange(1, array.shape[0] - 1):
        lap[i] = array[i + 1] - 2 * array[i] + array[i - 1]  # approximation of 1D laplacian (second positional derivative)
    return lap


@cuda.jit  # can run in parallel because of no call to previous element in laplacian array
def custom_laplacian_gpu(array, lap):
    i = cuda.grid(1)
    if 1 <= i < (array.shape[0] - 1):
        lap[i] = array[i + 1] - 2 * array[i] + array[i - 1]  # approximation of 1D laplacian (second positional derivative)
    else:
        lap[i] = 0


# def update(val):
#     time = int(time_slider.val)
#     l.set_ydata(wmaps_arr[int(time / time_step)])
#     fig.canvas.draw_idle()


init_time = time.time()
wmaps_arr = iterate(wmap, time_step, int(100 / time_step))
print('Elapsed time: ', time.time() - init_time)
# time_slider.on_changed(update)


def animate(frame):
    l.set_ydata(wmaps_arr[frame])
    return l,


plt.show()
frame_skip = 10 
frames = np.arange(0, len(wmaps_arr), frame_skip)
ani = FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)
writer = FFMpegWriter(fps=240, metadata=dict(artist='Alankar Sovani'), bitrate=1800)
ani.save('wave_simulation_fast.mp4', writer=writer)

