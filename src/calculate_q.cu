#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <cstdio>

#define BLOCK_DIM_32 32
#define NUM_THREADS BLOCK_DIM_32 * BLOCK_DIM_32

__global__ void calculate_a_kernel(double* a, double* x, double* Z, int m, int n)
{
     __shared__ double x_shared[BLOCK_DIM_32];
    __shared__ double M_Z[BLOCK_DIM_32][BLOCK_DIM_32];

    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_32;
    int tx = tid % BLOCK_DIM_32;
    // Rows that we are calculating the norm for
    int j = blockIdx.x * BLOCK_DIM_32 + tx;

    double val = 0.0;
    
    for (int iter = 0; iter < (n + BLOCK_DIM_32 - 1)/ BLOCK_DIM_32; ++iter)
    {
        // Collaborative loading of x and Z into shared memory
        int ind = iter * BLOCK_DIM_32 + tid;
        if (tid < BLOCK_DIM_32 && ind < n) x_shared[tid] = x[ind];
        else x_shared[tid] = 0.0;

        int row = j;
        int col = iter * BLOCK_DIM_32 + ty;
        if (row < m && col < n) M_Z[ty][tx] = Z[col * m + row];
        else M_Z[ty][tx] = 0.0;
        __syncthreads();

        // Calculate sum of norms
        if (j < m)
        {
            #pragma unroll
            for (int k = 0; k < BLOCK_DIM_32; ++k)
            {
                val += (x_shared[k] - M_Z[k][tx]) * (x_shared[k] - M_Z[k][tx]);
            }
        }
        __syncthreads();
    }
    if (j < m) a[j] = sqrt(val);
}

void calculate_q_memory(cudaStream_t* streams,
                        cublasHandle_t* handles,
                        arma::mat& X,
                        double* d_Z,
                        double* d_B,
                        double* d_Q,
                        int x_n,
                        int m,
                        int n,
                        int k
                    )
{

    #pragma omp parallel 
    {
        cudaStream_t stream = streams[omp_get_thread_num()];
        cublasHandle_t cublasH = handles[omp_get_thread_num()];

        double* d_a;
        double* d_x;
        CUDA_CHECK(cudaMallocAsync(&d_a, m * sizeof(double), stream));
        CUDA_CHECK(cudaMallocAsync(&d_x, n * sizeof(double), stream));

        #pragma omp for
        for (unsigned int i = 0; i < x_n; i++) {
            // Copy i-th row of X to device d_a
            arma::rowvec x = X.row(i);
            CUDA_CHECK(cudaMemcpyAsync(d_x, x.memptr(), n * sizeof(double), cudaMemcpyHostToDevice, stream));

            int m_blocks = (m + BLOCK_DIM_32 - 1) / BLOCK_DIM_32;
            dim3 grid_dim(m_blocks, m_blocks);
            dim3 block_dim(NUM_THREADS);
            calculate_a_kernel<<<grid_dim, block_dim, 0, stream>>>(d_a, d_x, d_Z, m, n);

            // Calculate Q row as a * B
            double alpha = 1.0;
            double beta = 0.0;
            CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_T, m, k, &alpha, d_B, m, d_a, 1, &beta, d_x, 1));
            // cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, 1, m, m, &alpha, d_a, 1, d_B, m, &beta, d_x, 1);
            // Copy d_x into Q i-th row (column-major)
            CUBLAS_CHECK(cublasDcopy(cublasH, k, d_x, 1, d_Q + i, x_n));
        }
        CUDA_CHECK(cudaFreeAsync(d_a, stream)); 
        CUDA_CHECK(cudaFreeAsync(d_x, stream));
    }
}

void calculate_q_file();