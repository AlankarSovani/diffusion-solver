import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import cuda
import time
from matplotlib.animation import FuncAnimation, FFMpegWriter


# Initialize parameters
steps = 10000
time_step = 0.01
grain = 4
x = np.arange(0, 1001, grain)
y = np.arange(0, 1001, grain)
X, Y = np.meshgrid(x, y)
hmap = np.zeros((len(x), len(y)), dtype=np.float32)
# f = lambda x: 2/(1+np.exp(-(x - 500)/10))-1
# hmap = f(X) * f(Y)  # Example 2D data
hmap = np.sin(np.pi * X * 3 / 1000) * np.sin(np.pi * Y * 3 / 1000)
k = 100  # Heat conductivity
# Create plots (I let AI decide what this would look like)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
cax = ax.pcolormesh(X, Y, hmap, shading='auto', cmap='hot')
# Create the slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(ax_slider, 'Time', 0, 100, valinit=0)
# Update function
def update(val):
    current_time = time_slider.val
    cax.set_array(hmaps_arr[int(current_time/time_step)].ravel())
    fig.canvas.draw_idle()


@cuda.jit
def custom_laplacian_gpu(array, lap): # does not compute the laplacian for the boundary since no heat is lost on the boundary.
    i, j = cuda.grid(2)
    if 1 <= i < array.shape[0] - 1 and 1 <= j < array.shape[1] - 1:
        top = array[i - 1, j]
        bottom = array[i + 1, j]
        left = array[i, j - 1]
        right = array[i, j + 1]
        lap[i, j] = (top + bottom + left + right - 4 * array[i, j])/(grain**2)
    elif i == 0 or i == array.shape[0] - 1 or j == 0 or j == array.shape[1] - 1:
        if i == 0:
            top = array[i + 1, j]
        else: 
            top = array[i - 1, j]
        if i == array.shape[0] - 1:
            bottom = array[i - 1, j]
        else:
            bottom = array[i + 1, j]
        if j == 0:
            left = array[i, j + 1]
        else:
            left = array[i, j - 1]
        if j == array.shape[1] - 1:
            right = array[i, j - 1]
        else:
            right = array[i, j + 1]
        lap[i, j] = (top + bottom + left + right - 4 * array[i, j])/(grain**2)
     # discrete difference laplacian


def iterate(hmap, time_step, num_steps): # this function is only called on compilation, so we can create variables inside the function
    new_hmap = hmap
    hmaps_arr = np.empty((num_steps + 1, *hmap.shape))
    hmaps_arr[0] = new_hmap
    for i in range(num_steps):
        threadsperblock = (16, 32)
        blockspergrid_x = (new_hmap.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
        blockspergrid_y = (new_hmap.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        lap = np.zeros(new_hmap.shape, dtype=np.float32)
        d_lap = cuda.to_device(lap)
        custom_laplacian_gpu[blockspergrid, threadsperblock](hmaps_arr[i], d_lap)
        lap = d_lap.copy_to_host()
        hmaps_arr[i + 1] = hmaps_arr[i] + time_step * lap * k
    return hmaps_arr


start = time.time() 
hmaps_arr = iterate(hmap, time_step, steps)
print('Elapsed time: ', time.time() - start)
# Attach the update function to the slider
time_slider.on_changed(update)



def animate(frame):
    cax.set_array(hmaps_arr[frame].ravel())
    return cax,


plt.show()
start_animate = time.time()
frame_skip = 10
frames = np.arange(0, len(hmaps_arr), frame_skip)
ani = FuncAnimation(fig, animate, frames=frames, interval=20, blit=True)
# writer = FFMpegWriter(
#     fps=120,
#     metadata=dict(artist='Me'),
#     bitrate=7200,
#     codec='h264_nvenc',
#     extra_args=['-preset', 'fast', '-pix_fmt', 'yuv420p']
# )
writer = FFMpegWriter(
    fps=120,
    metadata=dict(artist='Me'),
    bitrate=7200,
    codec='libx264',
    extra_args=['-pix_fmt', 'yuv420p']  # Ensures compatibility with most players
)
ani.save(filename="videos/heat.mp4", writer=writer)
print('Animation took: ', time.time() - start_animate, ' seconds')
