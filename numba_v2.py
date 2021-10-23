import numpy as np
from numba import cuda
import cupy as cp
from cupy import prof
import sys
from math import floor

@cuda.reduce
def sum_reduce(a, b):
    return a + b

@cuda.jit
def numba_l2_norm(x):
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        x[i] = x[i] * x[i]

def main():
    loops = int(sys.argv[1])
    
    x = np.random.random(5)
    d_x = cuda.to_device(x)

    threads_per_block = 64
    blocks_per_grid = 16

    with prof.time_range("numba", 0):
        numba_l2_norm[blocks_per_grid, threads_per_block](d_x)
        output = np.sqrt(sum_reduce(d_x))

    for _ in range(loops):
        with prof.time_range("numba_loop", 0):
            numba_l2_norm[blocks_per_grid, threads_per_block](d_x)
            output = np.sqrt(sum_reduce(d_x))

if __name__ == "__main__":
    sys.exit(main())