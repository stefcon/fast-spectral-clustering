#include "../lib/lambda-lanczos/include/lambda_lanczos/lambda_lanczos.hpp"
#include "../lib/optimal_k.hpp"
#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <vector>
#include <iostream>


using stdvec = std::vector<double>;
using stdvecvec = std::vector<std::vector<double>>;

static stdvecvec mat_to_std_vec(arma::mat &A) {
    stdvecvec V(A.n_rows);
    for (size_t i = 0; i < A.n_rows; ++i) {
        V[i] = arma::conv_to< stdvec >::from(A.row(i));
    };
    return V;
}

int find_optimal_k(arma::mat& X, int k, arma::vec& eigvals)
{
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    // Initialize cusolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    // Cublas constants (used in cublas calls)
    const double alpha = 1.0;
    const double beta = 0.0;

    int m = X.n_rows;
    int n = X.n_cols;

    Timer tim;
    startTime(&tim);

    // Calculate affinity matrix
    double* d_A, *d_X;
    CUDA_CHECK(cudaMalloc((void**)&d_A, m * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_X, m * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_X, X.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice));
    // Sample m matrix rows from X and copy to d_Z
    calculate_affinity_matrix_cuda(d_A, d_X, m, n);
    // Free unneded memory: d_Z (still on CPU)
    CUDA_CHECK(cudaFree(d_X));
    std::cout << "Finished calculating affinity matrix" << std::endl;

    // Calculate M_star
    double* d_M_star;
    double* d_ww;
    double* d_ones;
    CUDA_CHECK(cudaMalloc((void**) &d_ww, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**) &d_ones, m * sizeof(double)));

    // Initialize d_ones_m
    ones_cuda(d_ones, m);
    // ww = A_11 * ones_m
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, m, m, &alpha, d_A, m, d_ones, 1, &beta, d_ww, 1));

    // M_star = D_star_ * A_11 * D_star_
    pow_vec(d_ww, m, -0.5);
    gemv_diag(d_A, d_ww, m, m, MUL_LEFT_T);
    gemv_diag(d_A, d_ww, m, m, MUL_RIGHT_T);
    // A_11 now holds M_star (overwritten)
    d_M_star = d_A;

    // Free unneded memory: d_ones
    CUDA_CHECK(cudaFree(d_ones));
    std::cout << "Finished calculating M_star" << std::endl;


    // Find the eigendecomp of M_star
    // double* d_eigvals;
    // double* d_eigvecs;
    // double* d_W;
    // CUDA_CHECK(cudaMalloc((void**)&d_eigvals, k * sizeof(double)));
    // CUDA_CHECK(cudaMalloc((void**)&d_eigvecs, m * k * sizeof(double)));
    // CUDA_CHECK(cudaMalloc(&d_W, sizeof(double) * m));
    // eig_dsymx_cusolver(cusolverH, d_M_star, d_W, m, k, d_eigvals, d_eigvecs);
    // // d_M_star has been overwritten by eigenvectors, so we 
    // // can deallocoate it as well as d_W
    // CUDA_CHECK(cudaFree(d_M_star));
    // CUDA_CHECK(cudaFree(d_W));


    stdvecvec matrix = mat_to_std_vec(X);
    auto mv_mul = [&](const std::vector<double>& in, std::vector<double>& out) {
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                out[i] += matrix[i][j]*in[j];
            }
        }
    };


    /* Execute Lanczos algorithm */
    lambda_lanczos::LambdaLanczos<double> engine(mv_mul, m, true, k); // Find 3 maximum eigenvalue
    std::vector<double> eigenvalues;
    std::vector<std::vector<double>> eigenvectors;
    engine.run(eigenvalues, eigenvectors);
    std::cout << "Finished calculating eigenvalues" << std::endl;


    // Copy eigenvectors to host
    // Convert to arma::vec and arma::mat
    eigvals = arma::conv_to<arma::vec>::from(eigenvalues);
    // eigvecs = arma::conv_to<arma::mat>::from(eigenvectors);
    // CUDA_CHECK(cudaMemcpy(eigvals.memptr(), d_eigvals, k * sizeof(double), cudaMemcpyDeviceToHost));
    // // CUDA_CHECK(cudaMemcpy(eigvecs.memptr(), d_eigvecs, m * k * sizeof(double), cudaMemcpyDeviceToHost));
    // // Free unneded memory: d_eigvals, d_eigvecs
    // CUDA_CHECK(cudaFree(d_eigvals));
    // CUDA_CHECK(cudaFree(d_eigvecs));

    eigvals = eigvals / eigvals.max();
    eigvals.save("eigvals.csv", arma::csv_ascii);

    // Calculate the differnce between consecutive eigenvalues
    // and find the index of the largest difference
    // Divide eigvals by the largest eigenvalue
    arma::vec eigval_diffs = arma::abs(arma::diff(eigvals));
    int max_eigval_diff_idx = arma::index_max(eigval_diffs);

    stopTime(&tim);
    printElapsedTime(tim, "find_optimal_k");
    return max_eigval_diff_idx + 1;



}