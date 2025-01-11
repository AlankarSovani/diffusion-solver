import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from numba import njit
from numba import prange


# Initialize parameters
time_step = 0.1
x = np.linspace(0, 1000, int(10 / time_step))
y = np.linspace(0, 1000, int(10 / time_step))
X, Y = np.meshgrid(x, y)
hmap = np.sin(0.01*X)  # Example 2D data

# Create plots (I let AI decide what this would look like)
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
cax = ax.pcolormesh(X, Y, hmap, shading='auto', cmap='hot')

# Create the slider
ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
time_slider = Slider(ax_slider, 'Time', 0, 1000, valinit=0)

# Update function
def update(val):
    current_time = time_slider.val
    cax.set_array(hmaps_arr[int(current_time/time_step)].ravel())
    fig.canvas.draw_idle()

# naive implementation of the laplacian function

# def laplacian(array):
#     gx, gy = np.gradient(array)
#     gxx, gxy = np.gradient(gx)
#     gyx, gyy = np.gradient(gy)
#     lap = gxx + gyy
#     for i in range(len(lap)): # no heat is lost on the boundary
#         lap[0][i] = 0
#         lap[i][0] = 0
#         lap[-1][i] = 0
#         lap[i][-1] = 0
#     return lap


# slow implementation of the iterate function
# def iterate(hmap):
#     new_hmap = hmap.copy()
#     hmaps_arr = [new_hmap]
#     for i in range (int(1000/time_step)):
#         hmaps_arr.append(hmaps_arr[i] + time_step * laplacian(hmaps_arr[i]))
#     return hmaps_arr




@njit
def custom_gradient(array): # does not calculate the gradient on the boundary as no heat is lost on the boundary. saves computations.
    grad = np.zeros_like(array)
    for i in prange(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            grad[i, j] = (array[i + 1, j] - array[i - 1, j]) / 2, (array[i, j + 1] - array[i, j - 1]) / 2 # discrete difference gradient
    return grad

@njit
def custom_laplacian(array): # does not compute the laplacian for the boundary since no heat is lost on the boundary.
    lap = np.zeros_like(array)
    for i in prange(1, array.shape[0] - 1):
        for j in range(1, array.shape[1] - 1):
            lap[i, j] = array[i + 1, j] + array[i - 1, j] + array[i, j + 1] + array[i, j - 1] - 4 * array[i, j] # discrete difference laplacian
    return lap

@njit
def iterate(hmap, time_step, num_steps): # this function is only called on compilation, so we can create variables inside the function
    new_hmap = hmap.copy()
    hmaps_arr = np.empty((num_steps + 1, *hmap.shape))
    hmaps_arr[0] = new_hmap
    for i in prange(num_steps):
        lap = custom_laplacian(hmaps_arr[i]) # may call njit functions inside of an njit function
        hmaps_arr[i + 1] = hmaps_arr[i] + time_step * lap
        # No heat is lost on the boundary
        hmaps_arr[i + 1][0, :] = 0
        hmaps_arr[i + 1][:, 0] = 0
        hmaps_arr[i + 1][-1, :] = 0
        hmaps_arr[i + 1][:, -1] = 0
    return hmaps_arr


hmaps_arr = iterate(hmap, 0.1, 10000)
# Attach the update function to the slider
time_slider.on_changed(update)
plt.show()