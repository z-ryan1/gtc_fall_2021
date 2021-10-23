import numpy as np
from numpy import histogram
from numba import cuda
import sys
from cupy import prof
from scipy import signal
from serial_python_v1 import cpu_histogram

# Using the built in numpy histogram
def main():
    loops = int(sys.argv[1])
    
    data = np.random.random(2 ** 20)
    
    histogram_out = np.zeros(shape=1000, dtype=np.int32)
    histogram_out_numpy = np.zeros(shape=1000, dtype=np.int32)
    nbins = histogram_out.shape[0]

    xmin = np.float32(-1.0)
    xmax = np.float32(1.0)
    
    with prof.time_range("cpu histogram", 0):
        cpu_histogram(data, xmin, xmax, histogram_out)

    with prof.time_range("numpy built in", 0):
        histogram_out_numpy, bins = np.histogram(data, nbins, range=(-1.0,1.0))
    
    # Compare results
    np.testing.assert_allclose(
        histogram_out, histogram_out, 1e-3
    )

    for _ in range(loops):
        with prof.time_range("cpu histogram loop", 0):
            cpu_histogram(data, xmin, xmax, histogram_out)

    for _ in range(loops):
        with prof.time_range("built in loop", 0):
            histogram_out_numpy, bins = np.histogram(data, nbins, range=(-1.0,1.0))

if __name__ == "__main__":
    sys.exit(main())
