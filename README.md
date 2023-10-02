# Fast Spectral Clustering
Parallelized implementation of the algorithm proposed in the paper "Time and Space Efficient Spectral Clustering via Column Sampling" by Mu Li et al., 2011. done as my final bachelor thesis.

# Dependencies
Most of the dependencies that are required to run this code can be installed through running the scripts ```lib/lib.sh``` with sudo privileges.

In addtion, to run the parallelized implementation of the algorithm, host computer is required to have the CUDA and NVCC compiler installed ([Installation guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/contents.html)).

To install all the Python dependencies required to run the scripts found in the code base, user should create a [new Python virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) and install the dependencies found in the ```requirements.txt``` file.

# Building and running the project
To set up the project structure, ```lib/init.sh``` script should be run first. To download the data used for testing purposes, user should run ```lib/data.sh``` script.

To build the project, run command ```make default-target```. Main executable named ```program``` will be created, but to run it with the use of multiple OpenMP threads, it should be run using ```runner.sh``` script.

