import numpy as np
import sys
import cupy as cp
from cupy import prof, histogram, array
from scipy import signal
from numba import cuda


def main():
    loops = int(sys.argv[1])
    
    data = np.random.random(2 ** 20)
    d_data = array(data)

    histogram_out = cp.zeros(shape=1000, dtype=cp.int32)
    histogram_out_numpy = np.zeros(shape=1000, dtype=np.int32)
    nbins = histogram_out.shape[0]

    with prof.time_range("NumPy built in", 0):
        histogram_out, bins = np.histogram(data, nbins, range=(-1.0, 1.0))

    with prof.time_range("CuPy built in", 0):
        d_histogram_out, bins = cp.histogram(d_data, nbins, range=(-1.0, 1.0))

    # Compare results
    np.testing.assert_allclose(
        histogram_out, cp.asnumpy(d_histogram_out), 1e-3
    )

    for _ in range(loops):
        with prof.time_range("CuPy built in loop", 0):
            d_histogram_out, bins = cp.histogram(d_data, nbins, range=(-1.0, 1.0))

    for _ in range(loops):
        with prof.time_range("NumPy built in loop", 0):
            histogram_out_numpy, bins = np.histogram(data, nbins, range=(-1.0,1.0))

if __name__ == "__main__":
    sys.exit(main())                
