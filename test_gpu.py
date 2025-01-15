import numpy as np
from numba import njit, prange, cuda
import time

@cuda.jit
def test_function(array):
    # function code
    pos = cuda.grid(1)
    if pos < array.size:
        array[pos] += 1
    return

# gpu testing


def increment_by_one_nogpu(array):
    for i in range(array.size):
        array[i] += 1
    return array

def test():
    array = np.arange(1_000_000)
    start = time.time()
    increment_by_one_nogpu(array)
    print("Time taken for CPU: ", time.time() - start)
    array = np.arange(100_000_000)
    start = time.time()
    threadsperblock = 32
    blockspergrid = (array.size + (threadsperblock - 1)) // threadsperblock
    test_function[blockspergrid, threadsperblock](array)
    print("Time taken for GPU: ", time.time() - start)
    
test()