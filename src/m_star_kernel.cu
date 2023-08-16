#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <cstdio>

#define BLOCK_SIZE 16
#define NUM_THREADS 256


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

void test_calculate_m_star(cublasHandle_t& handle, arma::mat& A_11, arma::mat& M_star)
{
    // Transfering this code to cuda:
    // arma::vec ww = A_11 * arma::ones<arma::vec>(m);
    // arma::mat D_star_ = arma::diagmat(arma::pow(arma::sqrt(ww), -1));
    // arma::mat M_star = D_star_ * A_11 * D_star_;

    cublasStatus_t stat;
    // cublasHandle_t handle;
    int m = A_11.n_rows;
    Timer timer;
    // cublasCreate(&handle);

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

