#ifndef KERNELS_HPP
#define KERNELS_HPP

#define ARMA_ALLOW_FAKE_GCC
#define ARMA_USE_OPENMP
#include <armadillo>
#include <omp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

typedef enum {
    MUL_LEFT_T = 0,
    MUL_RIGHT_T = 1,
} vectorDiagMul_t;

extern void test_calculate_affinity_matrix(arma::mat& A_11, arma::mat& Z, double mu);

extern void calculate_affinity_matrix_cuda(double* d_A_11, double* d_Z, int m, int n);

void test_calculate_m_star( cublasHandle_t& handle, 
                            arma::mat& A_11,
                            arma::mat& M_star
                            );

void diagmat_cublas(cublasHandle_t handle, 
                    double* d_diag,
                    double* d_D,
                    int m
                    );

void pow_vec(double* d_ww, int m, double p);

void sqrt_vec(double* d_ww, int m);

void ones_cuda(double* d_ones, int m);

void gemv_cublas(   cublasHandle_t handle,
                    double* d_M,
                    double* d_v,
                    double* d_result,
                    int m,
                    int n
                );

void eig_dsymx_cusolver( 
                        cusolverDnHandle_t cusolverH,
                        double* d_A,
                        double* d_W,
                        int m,
                        int k, 
                        double* d_eigvals = nullptr, 
                        double* d_eigvecs = nullptr
                    );

void test_eig_kernel(arma::mat mat, int  m);

void orthogonalize_cuda(
    cublasHandle_t cublasH, 
    cusolverDnHandle_t cusolverH,
    double* d_U,
    double* d_Lam,
    int m,
    int n,
    double* d_res_U
    );

void calculate_q_memory(cudaStream_t* streams,
                        cublasHandle_t cublasH,
                        arma::mat& X,
                        double* d_Z,
                        double* d_B,
                        double* d_Q,
                        int x_n,
                        int m,
                        int n,
                        int k
                        );

void gemv_diag(double* d_M,
               double* d_v,
               int m,
               int n,
               vectorDiagMul_t type
               );


#endif