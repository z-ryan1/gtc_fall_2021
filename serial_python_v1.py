import numpy as np
import sys
from cupy import prof
from scipy import signal

# Naive serial implementation of cpu_histogram

# Pulled from numba NVIDIA cuda tutorial
def cpu_histogram(x, xmin, xmax, histogram_out):
    nbins = histogram_out.shape[0]
    bin_width = (xmax - xmin) / nbins
    
    for element in x:
        bin_number = np.int32(
            (element - xmin)/bin_width)
        if bin_number >= 0 and \
        bin_number < histogram_out.shape[0]:
            histogram_out[bin_number] += 1

def main():
    loops = int(sys.argv[1])
    
    data = np.random.random(2 ** 20)
    xmin = np.float32(-1.0)
    xmax = np.float32(1.0)
    histogram_out = np.zeros(shape=1000, dtype=np.int32)

    with prof.time_range("cpu histogram", 0):
        cpu_histogram(data, xmin, xmax, histogram_out)

    for _ in range(loops):
        with prof.time_range("cpu histogram loop", 0):
            cpu_histogram(data, xmin, xmax, histogram_out)

if __name__ == "__main__":
    sys.exit(main())
