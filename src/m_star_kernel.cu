#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <cstdio>

#define BLOCK_SIZE 16
#define NUM_THREADS 256
#define BLOCK_DIM_32 32
#define NUM_THREADS_32 1024


__global__ void initialize_with_ones(double* d_ones, int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        d_ones[i] = 1;
    }
}

__global__ void pow_kernel(double* d_ww, int m, double p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        d_ww[i] = pow(d_ww[i], p);
    }
}

__global__ void sqrt_kernel(double* d_ww, int m) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < m)
    {
        d_ww[i] = sqrt(d_ww[i]);
    }
}

__global__ void gemv_diag_left(double* d_M, double* d_v, int m, int n)
{
    __shared__ double x_shared[BLOCK_DIM_32];

    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_32;
    int tx = tid % BLOCK_DIM_32;
    // Rows that we are going to multiply
    int i = blockIdx.y * BLOCK_DIM_32 + ty;
    int j = blockIdx.x * BLOCK_DIM_32 + tx;

    // int ind = i * BLOCK_DIM_32 + ty;
    if (tx == 0 && i < m) x_shared[ty] = d_v[i];
    else x_shared[ty] = 0.0;
    __syncthreads();
    double elem = x_shared[ty];

    
    if (i < m && j < n)
    {
        d_M[j * m + i] *= elem;
    }
}

__global__ void gemv_diag_right(double* d_M, double* d_v, int m, int n)
{
    __shared__ double x_shared[BLOCK_DIM_32];

    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_32;
    int tx = tid % BLOCK_DIM_32;
    // Rows that we are going to multiply
    int i = blockIdx.y * BLOCK_DIM_32 + ty;
    int j = blockIdx.x * BLOCK_DIM_32 + tx;

    if (ty == 0 && j < n) x_shared[tx] = d_v[j];
    else x_shared[tx] = 0.0;
    __syncthreads();

    
    if (i < m && j < n)
    {
        d_M[j * m + i] *= x_shared[tx];
    }
}

void diagmat_cublas(cublasHandle_t handle, double* d_diag, double* d_D, int m)
{
    cudaMemset(d_D, 0, m * m * sizeof(double));
    CUBLAS_CHECK(cublasDcopy(handle, m, d_diag, 1, d_D, m + 1));
}

void pow_vec(double* d_ww, int m, double p)
{
    pow_kernel<<<(m + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(d_ww, m, p);
}

void sqrt_vec(double* d_ww, int m)
{
    sqrt_kernel<<<(m + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(d_ww, m);
}

void ones_cuda(double* d_ones, int m)
{
    initialize_with_ones<<<(m + NUM_THREADS - 1)/NUM_THREADS, NUM_THREADS>>>(d_ones, m);
}

void gemv_cublas(cublasHandle_t handle, double* d_M, double* d_v, double* d_result, int m, int n)
{
    const double alpha = 1;
    const double beta = 0;
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, m, n, &alpha, d_M, m, d_v, 1, &beta, d_result, 1));
}

void gemv_diag(double* d_M, double* d_v, int m, int n, vectorDiagMul_t type)
{
    // -----------------------------------------------------------------------------------------
    // Multiplies matrix d_M and vector d_v as if d_v was a diagonal matrix
    // Args:
    //      d_M: matrix of size m x n
    //      d_v: vector of size m/n (depending on type)
    //      m: number of rows of d_M
    //      n: number of columns of d_M
    //      type: position of the vector in regards to the matrix d_M (MUL_ROW_T or MUL_COL_T)
    // -----------------------------------------------------------------------------------------
    dim3 grid((n + BLOCK_DIM_32 - 1)/BLOCK_DIM_32, (m + BLOCK_DIM_32 - 1)/BLOCK_DIM_32);
    dim3 block(NUM_THREADS_32);
    if (type == MUL_LEFT_T)
    {
        gemv_diag_left<<<grid, block>>>(d_M, d_v, m, n);
    }
    else if (type == MUL_RIGHT_T)
    {
        gemv_diag_right<<<grid, block>>>(d_M, d_v, m, n);
    }
}

__global__ void parallel_subtract_mul_kernel(double* d_Z, double* d_a, double* d_x, double* d_result, int m, int n)
{
    // Subtract from every column of d_Z the vector d_x
    // Args:
    //      d_Z: matrix of size m x n
    //      d_a: vector of size m
    //      d_x: vector of size n
    //      m: number of rows of d_Z
    //      n: number of columns of d_Z
    int tid = threadIdx.x;
    int ty = tid / BLOCK_DIM_32;
    int tx = tid % BLOCK_DIM_32;
    // Rows that we are going to multiply
    int i = blockIdx.y * BLOCK_DIM_32 + ty;
    int j = blockIdx.x * BLOCK_DIM_32 + tx;
    double elem = 0.0;
    if (i < m && j < n)
    {
        elem = (d_x[j] - d_Z[j * m + i]) * (d_x[j] - d_Z[j * m + i]);
        d_result[j * m + i]= elem;
    }
}

void calculate_affinity_row_cuda(cublasHandle_t cublasH, double* d_Z, double* d_a, double* d_x, double* d_result, int m, int n)
{
    // Calculate affinity row for every point in Z

    // double *d_x, *d_ones, *d_result;
    double alpha = 1.0;
    double beta = 0.0;
    // CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));
    // CUDA_CHECK(cudaMalloc((void**)&d_ones, n * sizeof(double)));
    // CUDA_CHECK(cudaMalloc((void**)&d_result, m * n * sizeof(double)));
    // CUDA_CHECK(cudaMemcpy(d_x, x.memptr(), n * sizeof(double), cudaMemcpyHostToDevice));


    dim3 grid((n + BLOCK_DIM_32 - 1)/BLOCK_DIM_32, (m + BLOCK_DIM_32 - 1)/BLOCK_DIM_32);
    dim3 block(NUM_THREADS_32);
    parallel_subtract_mul_kernel<<<grid, block>>>(d_Z, d_a, d_x, d_result, m, n);

    // Change d_x to vector of ones (not needed anymore!)
    ones_cuda(d_x, n);
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, m, n, &alpha, d_result, m, d_x, 1, &beta, d_a, 1));
    sqrt_vec(d_a, m);

    // CUDA_CHECK(cudaFree(d_x));
    // CUDA_CHECK(cudaFree(d_result));
}

void test_calculate_m_star(cublasHandle_t& handle, arma::mat& A_11, arma::mat& M_star)
{
    // Transfering this code to cuda:
    // ------------------------------
    // arma::vec ww = A_11 * arma::ones<arma::vec>(m);
    // arma::mat D_star_ = arma::diagmat(arma::pow(arma::sqrt(ww), -1));
    // arma::mat M_star = D_star_ * A_11 * D_star_;
    // ------------------------------
    cublasStatus_t stat;
    Timer timer;
    
    int m = A_11.n_rows;
    
    double* d_A_11;
    double* d_ww;
    double* d_ones_m;
    double* d_D_star;
    double* d_DA;
    double* d_M_star;

    // Allocate memory on device
    startTime(&timer);
    cudaMalloc((void**) &d_A_11, m * m * sizeof(double));
    cudaMalloc((void**) &d_ww, m * sizeof(double));
    cudaMalloc((void**) &d_ones_m, m * sizeof(double));
    cudaMalloc((void**) &d_D_star, m * m * sizeof(double));
    cudaMalloc((void**) &d_DA, m * m * sizeof(double));
    cudaMalloc((void**) &d_M_star, m * m * sizeof(double));

    // Initialize memory on device
    ones_cuda(d_ones_m, m);
    cudaMemcpy(d_A_11, A_11.memptr(), m * m * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(d_D_star, 0, m * m * sizeof(double));
    stopTime(&timer);
    printElapsedTime(timer, "Time to allocate and initialize memory on device: ");

    // Calculate ww = A_11 * ones(m)
    startTime(&timer);
    const double alpha = 1;
    const double beta = 0;
    stat = cublasDgemv(handle, CUBLAS_OP_N, m, m, &alpha, d_A_11, m, d_ones_m, 1, &beta, d_ww, 1);
    if (stat == CUBLAS_STATUS_EXECUTION_FAILED)
    {
        printf("Error in cublasDgemv\n");
    }


    // // Calculate D_star
    pow_vec(d_ww, m, -0.5);
    diagmat_cublas(handle, d_ww, d_D_star, m);

    // // Calcualte M_star
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, d_D_star, m, d_A_11, m, &beta, d_DA, m);
    if (stat == CUBLAS_STATUS_EXECUTION_FAILED)
    {
        printf("Error in cublasDgemm 1\n");
    }
    stat = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, m, m, &alpha, d_DA, m, d_D_star, m, &beta, d_M_star, m);
    if (stat == CUBLAS_STATUS_EXECUTION_FAILED)
    {
        printf("Error in cublasDgemm 2\n");
    }
    cudaDeviceSynchronize();
    stopTime(&timer);
    printElapsedTime(timer, "Time to calculate M_star: ", GREEN);


    // Copy data from device
    startTime(&timer);
    cudaMemcpy(M_star.memptr(), d_M_star, m * m * sizeof(double), cudaMemcpyDeviceToHost);

    // Free memory on device and destroy cublas context
    // cublasDestroy(handle);
    cudaFree(d_A_11);
    cudaFree(d_ww);
    cudaFree(d_ones_m);
    cudaFree(d_D_star);
    cudaFree(d_DA);
    cudaFree(d_M_star);
    stopTime(&timer);
    printElapsedTime(timer, "Time to copy data from device and free memory: ");
}

