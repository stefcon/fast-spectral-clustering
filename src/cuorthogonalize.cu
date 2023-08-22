#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <cstdio>


void orthogonalize_cuda(
    cublasHandle_t cublasH, 
    cusolverDnHandle_t cusolverH,
    double* d_U,
    double* d_Lam,
    int m,
    int n,
    double* d_res_U
    )
{
    // ---------------------------------------------------------------
    // Orthogonalization algorithm proposed in "Fast Spectral Clustering" by M. Li et al.
    // ---------------------------------------------------------------
    // Input: d_U (m x n), d_Lam (n x n)
    // Output: d_res_U (m x n)
    // ---------------------------------------------------------------
    // Armadillo implementation:
    //
    // arma::mat P = U.t() * U;
    // arma::vec Sig;
    // arma::mat Vp;
    // arma::eig_sym(Sig, Vp, P);
    //
    // arma::mat Sig_ = arma::diagmat(arma::sqrt(Sig));
    // B = Sig_ * Vp.t() * Lam * Vp * Sig_;
    // arma::vec Lam_tilde;
    // arma::mat V_tilde;
    // arma::eig_sym(Lam_tilde, V_tilde, B);
    // U = U * Vp * arma::diagmat(arma::pow(arma::sqrt(Sig), -1)) * V_tilde;
    // ---------------------------------------------------------------
    
    double* d_P;
    double* d_Sig;
    double* d_Vp;
    double* d_Sig_;
    double* d_B;
    double* d_Lam_tilde;
    double* d_V_tilde;

    double* d_m_n_tmp1, * d_m_n_tmp2;
    double* d_n_n_tmp1, * d_n_n_tmp2;

    const double one = 1.0;
    const double zero = 0.0;

    CUDA_CHECK(cudaMalloc((void**)&d_P, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Sig, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Sig_, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_B, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Lam_tilde, n * sizeof(double)));

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &one, d_U, m, d_U, m, &zero, d_P, n));
    eig_dsymx_cusolver(cusolverH, d_P, d_Sig, n, n);
    // *** d_P is destroyed during the function and filled with its eigenvectors after function call ***
    d_Vp = d_P;


    sqrt_vec(d_Sig, n);
    diagmat_cublas(cublasH, d_Sig, d_Sig_, n);

    // B = Sig_ * Vp.t() * Lam * Vp * Sig_;
    // CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, d_Sig_, n, d_Vp, n, &zero, d_n_n_tmp1, n));
    // CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_Lam, n, d_Vp, n, &zero, d_n_n_tmp2, n));
    // CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_n_n_tmp1, n, d_n_n_tmp2, n, &zero, d_n_n_tmp3, n));
    // CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_n_n_tmp3, n, d_Sig_, n, &zero, d_B, n));

    CUDA_CHECK(cudaMalloc((void**)&d_n_n_tmp1, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_n_n_tmp2, n * n * sizeof(double)));

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T, n, n, n, &one, d_Sig_, n, d_Vp, n, &zero, d_n_n_tmp1, n));
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_n_n_tmp1, n, d_Lam, n, &zero, d_n_n_tmp2, n));
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_n_n_tmp2, n, d_Vp, n, &zero, d_n_n_tmp1, n));
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, d_n_n_tmp1, n, d_Sig_, n, &zero, d_B, n));

    CUDA_CHECK(cudaFree(d_n_n_tmp1));
    CUDA_CHECK(cudaFree(d_n_n_tmp2));


    eig_dsymx_cusolver(cusolverH, d_B, d_Lam_tilde, n, n);
    // *** d_B is destroyed during the function and filled with its eigenvectors after function call ***
    d_V_tilde = d_B;


    // U = U * Vp * arma::diagmat(arma::pow(arma::sqrt(Sig), -1)) * V_tilde;
    pow_vec(d_Sig, n, -1);
    diagmat_cublas(cublasH, d_Sig, d_Sig_, n);

    CUDA_CHECK(cudaMalloc((void**)&d_m_n_tmp1, m * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_m_n_tmp2, m * n * sizeof(double)));

    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, d_U, m, d_Vp, n, &zero, d_m_n_tmp1, m));
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, d_m_n_tmp1, m, d_Sig_, n, &zero, d_m_n_tmp2, m));
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, m, n, n, &one, d_m_n_tmp2, m, d_V_tilde, n, &zero, d_res_U, m));

    // Free memory
    CUDA_CHECK(cudaFree(d_Sig));
    CUDA_CHECK(cudaFree(d_Vp));
    CUDA_CHECK(cudaFree(d_Sig_));
    CUDA_CHECK(cudaFree(d_Lam_tilde));
    CUDA_CHECK(cudaFree(d_V_tilde));
    CUDA_CHECK(cudaFree(d_m_n_tmp1));
    CUDA_CHECK(cudaFree(d_m_n_tmp2));
}