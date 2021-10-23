#!/bin/bash
declare LOOPS=$1

if [ -z "$LOOPS" ];
then
	LOOPS=10
else
	LOOPS=${LOOPS}
fi

#Test serial Python naive histogram

echo -e "**************************************************"
echo -e "Test serial_python_v1.py ${LOOPS}"
echo -e "Naive histogram"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 serial_python_v1.py ${LOOPS}
echo -e


#Test NumPy histogram

echo -e "**************************************************"
echo -e "Test serial_python_v2.py ${LOOPS}"
echo -e "NumPy histogram"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 serial_python_v2.py ${LOOPS}
echo -e

# Test Numba histogram w/ rounding

echo -e "**************************************************"
echo -e "Test numba_v1.py ${LOOPS}"
echo -e "Numba custom histogram kernel, and forall rounding"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 numba_v1.py ${LOOPS}
echo -e

# Test NumPy and CuPy built in implementation of histogram

echo -e "**************************************************"
echo -e "Test cupy_v1.py ${LOOPS}"
echo -e "NumPy and CuPy built in implementation of histogram"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_v1.py ${LOOPS}
echo -e


# Test Numba reduce generator for l2 norm calculation

echo -e "**************************************************"
echo -e "Test numba_v2.py ${LOOPS}"
echo -e "Numba reduce generator for l2 norm"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 numba_v2.py ${LOOPS}
echo -e

# Test CuPy reduction kernel for l2 norm calculation

echo -e "**************************************************"
echo -e "Test cupy_v2.py ${LOOPS}"
echo -e "CuPy reduction kernel for l2 norm"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_v2.py ${LOOPS}
echo -e

# Test CuPy and Numba for CuPy array as input to custom kernel

echo -e "**************************************************"
echo -e "Test cupy_and_numba_v1.py ${LOOPS}"
echo -e "CuPy array as input to custom kernel"
echo -e "**************************************************"
nsys profile --sample=none --trace=cuda,nvtx --stats=true python3 cupy_and_numba_v1.py ${LOOPS}
echo -e
