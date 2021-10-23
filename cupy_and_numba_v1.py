import numpy as np
import sys
from cupy import prof
from scipy import signal
import cupy as cp
from numba import cuda
from cupyx.scipy import sparse

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
    # Create a random CuPy device array
    x = cp.random.random(2 ** 20)

    threads_per_block = 64
    blocks_per_grid = 16
    
    # CuPy device array as custom Numba kernel input
    with prof.time_range("numba", 0):
        numba_l2_norm[blocks_per_grid, threads_per_block](x)
        output = np.sqrt(sum_reduce(x))

if __name__ == "__main__":
    sys.exit(main())
