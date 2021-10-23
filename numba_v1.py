import numpy as np
from numba import cuda
import cupy as cp
from cupy import prof
import sys
from math import floor
from serial_python_v1 import cpu_histogram

# Condensed round function implemented in cuDF
@cuda.jit
def numba_round(in_col, out_col, decimal):
    i = cuda.grid(1)
    f = 10 ** decimal

    if i < in_col.size:
        ret = in_col[i] * f
        y = floor(ret)
        r = ret - y 
        if r > 0.5:
            y += 1.0 
        if r == 0.5:
            r = y - 2.0 * floor(0.5 * y)
            if r == 1.0:
                y += 1.0 
        tmp = y / f
        out_col[i] = tmp

# Pulled from numba NVIDIA cuda tutorial
@cuda.jit
def numba_histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    start = cuda.grid(1)
    stride = cuda.gridsize(1)

    for i in range(start, x.shape[0], stride):
        bin_number = np.int32((x[i] - xmin)/bin_width)
        if bin_number >= 0 and bin_number < histogram_out.shape[0]:
            cuda.atomic.add(histogram_out, bin_number, 1)

def main():
    loops = int(sys.argv[1])
    
    data = np.random.random(2 ** 20)
    in_data = cuda.to_device(data)
    #d_out_data = cuda.device_array_like(in_data)
    nelems = len(in_data)

    out_data = np.round(data, 3)
    numba_round.forall(nelems)(in_data, d_out_data, 3)
    
    xmin = np.float32(-1.0)
    xmax = np.float32(1.0)

    histogram_out = np.zeros(shape=1000, dtype=np.int32)
    d_histogram_out = cuda.to_device(histogram_out)
    
    threads_per_block = 64
    blocks_per_grid = (d_histogram_out.size + (threads_per_block - 1)) // threads_per_block

    with prof.time_range("cpu histogram", 0):
        cpu_histogram(data, xmin, xmax, histogram_out)

    with prof.time_range("numba", 0):
        numba_histogram[blocks_per_grid, threads_per_block](in_data, xmin, xmax, d_histogram_out)

    # Compare results
    np.testing.assert_allclose(
        histogram_out, d_histogram_out, 1e-3
    )

    for _ in range(loops):
        with prof.time_range("numba_loop", 0):
            numba_histogram[blocks_per_grid, threads_per_block](in_data, xmin, xmax, d_histogram_out)

    for _ in range(loops):
        with prof.time_range("cpu_loop", 0):
            cpu_histogram(data, xmin, xmax, histogram_out)

if __name__ == "__main__":
    sys.exit(main()) 
                                                                                                      