import numpy as np
import sys
import cupy as cp
from cupy import prof, histogram
from scipy import signal
from numba import cuda
from numpy.linalg import norm
from numba_v2 import *


_l2norm_kernel = cp.ReductionKernel(
    'T x',  # input params
    'T y',  # output params
    'x * x',  # map
    'a + b',  # reduce
    'y = sqrt(a)',  # post-reduction map
    '0',  # identity value
    'l2norm'  # kernel name
)

def l2norm(x):
    return _l2norm_kernel(x)

def main():
    loops = int(sys.argv[1])
        
    x = np.random.random(2 ** 20)
    d_x = cuda.to_device(x)

    threads_per_block = 64
    blocks_per_grid = (d_x.size + (threads_per_block - 1)) // threads_per_block

    with prof.time_range("cupy", 0):
        out = l2norm(d_x)

    with prof.time_range("numba", 0):
        numba_l2_norm[blocks_per_grid, threads_per_block](d_x)
        output = np.sqrt(sum_reduce(d_x))

    #Compare results
    np.testing.assert_allclose(
        cp.asnumpy(output), cp.asnumpy(out), 1e-3
    )

    for _ in range(loops):
        with prof.time_range("cupy_loop", 0):
            out_2 = l2norm(d_x)
            cp.cuda.runtime.deviceSynchronize()

    for _ in range(loops):
        with prof.time_range("numba_loop", 0):
            numba_l2_norm[blocks_per_grid, threads_per_block](d_x)
            output = np.sqrt(sum_reduce(d_x))

if __name__ == "__main__":
    sys.exit(main())                
