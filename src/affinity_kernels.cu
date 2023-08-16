#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include <cstdio>

#define BLOCK_DIM_32 32
#define NUM_THREADS 1024

__global__ void calculate_mu_kernel_reduction(double* Z, int m, int n, double * mu)
{
    // Use shared memory reduction for calculating mu
    // To calculate mu, it is needed to sum all the distances (norms) between
    // the points (rows) of Z (Z[i] and Z[j] for every i and j)
    // mu is array for every sub result of block reduction (gridDim.x * gridDim.x)
    // Further optimization could be done regarding collaborative loading of Z,
    // but it is not necessary since this kernel is not bottleneck based on the dimensions we
    // are dealing here (could be optmized for bigger samples of matrices)

    __shared__ double Mi[BLOCK_DIM_32][BLOCK_DIM_32];
    __shared__ double Mj[BLOCK_DIM_32][BLOCK_DIM_32];
    __shared__ double reduction_s[NUM_THREADS];

    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_32;
    int tx = tid % BLOCK_DIM_32;
    // Rows that we are calculating the norm for
    int i = blockIdx.y * BLOCK_DIM_32 + ty;
    int j = blockIdx.x * BLOCK_DIM_32 + tx;

    reduction_s[tid] = 0.0;
    
    for (int iter = 0; iter < (n + BLOCK_DIM_32 - 1)/ BLOCK_DIM_32; ++iter)
    {
        // Collaborative loading of Z into shared memory
        int row = i; // 1
        int col = iter * BLOCK_DIM_32 + tx;
        if (row < m && col < n) Mi[ty][tx] = Z[col * m + row];
        else Mi[ty][tx] = 0.0;

        row = j; // 1
        col = iter * BLOCK_DIM_32 + ty;
        if (row < m && col < n) Mj[ty][tx] = Z[col * m + row];
        else Mj[ty][tx] = 0.0;
        __syncthreads();

        // Calculate sum of norms
        if (i < m && j < m)
        {
            #pragma unroll
            for (int k = 0; k < BLOCK_DIM_32; ++k)
            {
                reduction_s[tid] += (Mi[ty][k] - Mj[k][tx]) * (Mi[ty][k] - Mj[k][tx]);
            }
        }
        __syncthreads();
    }
    // Shared memory reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            reduction_s[tid] += reduction_s[tid + s];
        __syncthreads();
    }

    if (tid == 0) mu[blockIdx.x * gridDim.x + blockIdx.y] = reduction_s[0];
}

__global__ void calculate_affinity_matrix_kernel(double* A_11, double* Z, double* mu_arr, int m, int n)
{
    __shared__ double sdata[NUM_THREADS];

    int i = blockIdx.y * BLOCK_DIM_32 + threadIdx.x / BLOCK_DIM_32;
    int j = blockIdx.x * BLOCK_DIM_32 + threadIdx.x % BLOCK_DIM_32;

    // Sum up mu values from all blocks
    // First, load every elem form mu_arr to shared memory, and then reduce it
    // in every thread to local variable mu
    double mu = 0.0;
    // double mu_temp = 0.0;
    // for (int iter = 0; iter < gridDim.x * gridDim.x; ++iter) mu += mu_arr[iter];
    int mu_arr_size = gridDim.x * gridDim.x;
    for (int iter = 0; iter < (mu_arr_size + blockDim.x - 1)/blockDim.x; ++iter)
    {
        if (threadIdx.x + iter*blockDim.x < mu_arr_size) sdata[threadIdx.x] = mu_arr[threadIdx.x  + iter*blockDim.x];
        else sdata[threadIdx.x] = 0.0;
        __syncthreads();
        for (int s = blockDim.x / 2; s > 0; s >>= 1)
        {
            if (threadIdx.x < s)
                sdata[threadIdx.x] += sdata[threadIdx.x + s];
            __syncthreads();
        }
        mu += sdata[0];
    }
    mu /= pow(mu, 2);
    mu = 1 / mu;

    if (i < m && j < m && i <= j)
    {
        double sum = 0.0;
        #pragma unroll
        for (int l = 0; l < n; l++)
        {
            sum += (Z[l * m + i] - Z[l * m + j]) * (Z[l * m + i] - Z[l * m + j]);
        }
        A_11[j * m + i] = exp(-sum * mu);
        A_11[i * m + j] = exp(-sum * mu);
    }
}

void test_calculate_affinity_matrix(arma::mat& A_11, arma::mat& Z, double mu)
{
    Timer timer;
    int n = Z.n_cols;
    int m = A_11.n_rows;
    int m_blocks = (m + BLOCK_DIM_32 - 1) / BLOCK_DIM_32;

    // Allocate memory on device
    double* d_A_11;
    double* d_Z;
    double* d_mu;

    startTime(&timer);
    // Allocate memory on device
    cudaMalloc((void**)&d_A_11, m * m * sizeof(double));
    cudaMalloc((void**)&d_Z, m * n * sizeof(double));
    cudaMalloc((void**)&d_mu, m_blocks * m_blocks * sizeof(double));

    // Copy data to device (affinity and Z matrix)
    // cudaMemcpy(d_A_11, A_11.memptr(), m * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_A_11, 0, m * m * sizeof(double));
    cudaMemcpy(d_Z, Z.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Memory allocation and data transfer to device");

    // Calculate affinity matrix
    startTime(&timer);
    // dim3 grid2(m_blocks, n_blocks);
    dim3 grid(m_blocks, m_blocks);
    dim3 block(NUM_THREADS);
    calculate_mu_kernel_reduction<<<grid, block>>>(d_Z, m, n, d_mu);
    calculate_affinity_matrix_kernel<<<grid, block>>>(d_A_11, d_Z, d_mu, m, n);
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Calculate affinity matrix", GREEN);

    // Copy data back to host
    startTime(&timer);
    cudaMemcpy(A_11.memptr(), d_A_11, m * m * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on device
    cudaFree(d_A_11);
    cudaFree(d_Z);
    cudaFree(d_mu);
    stopTime(&timer);
    printElapsedTime(timer, "Data transfer to host and memory deallocation");
}

void calculate_affinity_matrix(double* d_A_11, double* d_Z, int m, int n)
{
    int m_blocks = (m + BLOCK_DIM_32 - 1) / BLOCK_DIM_32;
    double* d_mu;
    cudaMalloc((void**)&d_mu, m_blocks * m_blocks * sizeof(double));
    cudaMemset(d_mu, 0,  m_blocks * m_blocks * sizeof(double));

    dim3 grid(m_blocks, m_blocks);
    dim3 block(NUM_THREADS);
    calculate_mu_kernel_reduction<<<grid, block>>>(d_Z, m, n, d_mu);
    calculate_affinity_matrix_kernel<<<grid, block>>>(d_A_11, d_Z, d_mu, m, n);
    cudaFree(d_mu);
}