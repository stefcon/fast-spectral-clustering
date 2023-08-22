#include "../lib/cssc.hpp"
#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <iostream>
#include <cstdio>
#include <vector>


using namespace std;

void CSSC::fit()
{
    Timer tim;
    startTime(&tim);
    arma::wall_clock timer;
    timer.tic();
    int x_n = X.n_rows;
    // Sample m points from X (Armadillo)
    std::vector<int> res = sample_without_replacement(0, X.n_rows - 1, m);
    arma::uvec inds = arma::conv_to<arma::uvec>::from(res);

    // Calculate Gaussian kernel
    arma::mat Z = X.rows(inds);
    double mu = 0.0;
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < m; j++) {
            mu += pow(arma::norm(Z.row(i) - Z.row(j)), 2);
        }
    }
    mu /= pow(m, 2);
    mu = 1 / mu;


    // Calculate the affinity matrix
    arma::mat A_11(m, m);
    for (unsigned int i= 0; i < m; i++) {
        for (unsigned int j=i; j < m; j++) {
            double val = exp(-mu * pow(arma::norm(Z.row(i) - Z.row(j)), 2));
            A_11(i, j) = val;
            A_11(j, i) = val;
        }
    }
    // A_11.save("A_11_cpu.csv", arma::csv_ascii);

    arma::vec ww = A_11 * arma::ones<arma::vec>(m);
    arma::mat D_star = arma::diagmat(ww); // Don't see why it's needed
    arma::mat D_star_ = arma::diagmat(arma::pow(arma::sqrt(ww), -1));
    arma::mat M_star = D_star_ * A_11 * D_star_;
    // M_star.save("M_star_cpu.csv", arma::csv_ascii);

    // Find the eigendecomposition of M_star
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M_star);
    
    eigval = eigval.rows(m-k, m-1);
    eigvec = eigvec.cols(m-k, m-1);
    
    arma::mat Lam = arma::diagmat(eigval);
    arma::mat B = D_star_ * eigvec * arma::diagmat(arma::pow(eigval, -1));
    // B.save("B_cpu.csv", arma::csv_ascii);
    arma::mat Q(x_n, k);
    for (unsigned int i = 0; i < x_n; i++) {
        arma::rowvec a(m);
        arma::rowvec x = X.row(i);
        for (unsigned int j = 0; j < m; j++)
        {
            a.col(j) = arma::norm(x - Z.row(j));
        }
        Q.row(i) = a * B;
    }
    
    // Save Q
    // Q.save("Q_cpu.csv", arma::csv_ascii);
    
    arma::vec dd = Q * Lam * Q.t() * arma::ones<arma::vec>(x_n);
    dd = arma::pow(arma::sqrt(dd), -1);
    arma::mat D_hat_ = arma::diagmat(dd);
    arma::mat U = D_hat_ * Q; // x_n x k

    // Orthogonalize U
    arma::mat P = U.t() * U; // k x k
    arma::vec Sig;
    arma::mat Vp;
    arma::eig_sym(Sig, Vp, P);
    
    arma::mat Sig_ = arma::diagmat(arma::sqrt(Sig)); // k x k
    B = Sig_ * Vp.t() * Lam * Vp * Sig_;
    arma::vec Lam_tilde;
    arma::mat V_tilde;
    arma::eig_sym(Lam_tilde, V_tilde, B);
    
    U = U * Vp * arma::diagmat(arma::pow(arma::sqrt(Sig), -1)) * V_tilde;

    // Cluster the approximated eigenvectors, U
    arma::mat centroids;
    arma::uvec y_hat(x_n);  
    bool status = arma::kmeans(centroids, U.t(), k, arma::random_subset, 10, false);
    if (!status) {
        std::cout << "Clustering failed!" << std::endl;
        this->y_hat = y_hat;
    }
    centroids = centroids.t();
    arma::vec d(k);
    stopTime(&tim);
    printElapsedTime(tim, "CPU", CYAN);

    for (unsigned int i = 0; i < x_n; i++) {
        for (unsigned int j = 0; j < k; j++) {
            d.row(j) = arma::norm(U.row(i) - centroids.row(j));
        }
        y_hat.row(i) = d.index_min();
    }

    this->y_hat = y_hat;
}

void CSSC::sample_matrix_X(arma::mat& Z, double *d_Z, int n)
{
    std::vector<int> res = sample_without_replacement(0, X.n_rows - 1, m);
    arma::uvec inds = arma::conv_to<arma::uvec>::from(res);
    Z = X.rows(inds);
    CUDA_CHECK(cudaMemcpy(d_Z, Z.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice));
}

void CSSC::calculate_affinity_matrix_A(double* d_A_11, double* d_Z, int m, int n)
{
    calculate_affinity_matrix_cuda(d_A_11, d_Z, m, n);
}

void CSSC::calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q)
{
    arma::mat Q(x_n, k);
    #pragma omp parallel for
    for (unsigned int i = 0; i < x_n; i++) {
        arma::rowvec a(m);
        arma::rowvec x = X.row(i);
        for (unsigned int j = 0; j < m; j++)
        {
            a.col(j) = arma::norm(x - Z.row(j));
        }
        Q.row(i) = a * B;
    }
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
}

void CSSC::gpu_fit(cublasHandle_t cublaH, cusolverDnHandle_t cusolveH)
{
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    // Initialize cusolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    // Cublas constants (used in cublas calsl)
    const double alpha = 1.0;
    const double beta = 0.0;

    Timer tim;
    startTime(&tim);

    // Calculate affinity matrix
    double* d_A_11;
    double* d_Z;
    CUDA_CHECK(cudaMalloc((void**)&d_A_11, m * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z, m * n * sizeof(double)));
    // Sample m matrix rows from X and copy to d_Z
    arma::mat Z(m, n);
    sample_matrix_X(Z, d_Z, n);
    calculate_affinity_matrix_A(d_A_11, d_Z, m, n);
    // Free unneded memory: d_Z (still on CPU)
    CUDA_CHECK(cudaFree(d_Z));

    // Calculate M_star
    double* d_M_star;
    double* d_ww;
    double* d_ones;
    CUDA_CHECK(cudaMalloc((void**) &d_ww, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**) &d_ones, m * sizeof(double)));

    // Initialize d_ones_m
    ones_cuda(d_ones, m);
    // ww = A_11 * ones_m
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, m, m, &alpha, d_A_11, m, d_ones, 1, &beta, d_ww, 1));

    // M_star = D_star_ * A_11 * D_star_
    pow_vec(d_ww, m, -0.5);
    gemv_diag(d_A_11, d_ww, m, m, MUL_LEFT_T);
    gemv_diag(d_A_11, d_ww, m, m, MUL_RIGHT_T);
    // A_11 now holds M_star (overwritten)
    d_M_star = d_A_11;

    // Free unneded memory: d_ones
    CUDA_CHECK(cudaFree(d_ones));


    // Find the eigendecomp of M_star
    double* d_eigvals;
    double* d_eigvecs;
    double* d_W;
    CUDA_CHECK(cudaMalloc((void**)&d_eigvals, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_eigvecs, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, sizeof(double) * m));
    eig_dsymx_cusolver(cusolverH, d_M_star, d_W, m, k, d_eigvals, d_eigvecs);
    // d_M_star has been overwritten by eigenvectors, so we 
    // can deallocoate it as well as d_W
    CUDA_CHECK(cudaFree(d_M_star));
    CUDA_CHECK(cudaFree(d_W));

    // Calculate B
    double* d_B;
    double* d_Lam;
    CUDA_CHECK(cudaMalloc((void**)&d_B, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Lam, k * k * sizeof(double)));

    // Diagonalize eigvals
    diagmat_cublas(cublasH, d_eigvals, d_Lam, k);
    
    // B = D_star_ (ww ^-1/2 diagonalized) * eigvecs * pow(eigvals, -1)
    gemv_diag(d_eigvecs, d_ww, m, k, MUL_LEFT_T);
    pow_vec(d_eigvals, k, -1);
    gemv_diag(d_eigvecs, d_eigvals, m, k, MUL_RIGHT_T);
    d_B = d_eigvecs;
    // Free unneded memory: d_eigvals
    CUDA_CHECK(cudaFree(d_eigvals));

    // Copy B to host arma::mat
    arma::mat B(m, k);
    CUDA_CHECK(cudaMemcpy(B.memptr(), d_B, m * k * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));

    // Calcualte Q on cpu 
    arma::mat Q(x_n, k);
    double* d_Q;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, x_n * k * sizeof(double)));
    calculate_affinity_Q(Z, B, n, d_Q);

    double* d_ones_xn;
    double* d_dd;
    double* d_QLam;
    double* d_Qt_ones;
    double* d_U;
    CUDA_CHECK(cudaMalloc((void**)&d_ones_xn, x_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_dd, x_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_QLam, x_n * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Qt_ones, k * sizeof(double)));

    // Initialize d_ones_nx
    ones_cuda(d_ones_xn, x_n);
    // dd = Q * Lam * Q.t() * d_ones_nx
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, x_n, k, k, &alpha, d_Q, x_n, d_Lam, k, &beta, d_QLam, x_n));

    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_T, x_n, k, &alpha, d_Q, x_n, d_ones_xn, 1, &beta, d_Qt_ones, 1));
    
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, x_n, k, &alpha, d_QLam, x_n, d_Qt_ones, 1, &beta, d_dd, 1));
    

    // U = diagmat(pow(dd, -0.5)) * Q
    pow_vec(d_dd, x_n, -0.5);
    gemv_diag(d_Q, d_dd, x_n, k, MUL_LEFT_T);
    // Q now holds U (overwritten)
    d_U = d_Q;
    
    // Free unneded memory: d_dd, d_ones_xn, d_QLam, d_Qt_ones
    CUDA_CHECK(cudaFree(d_dd));
    CUDA_CHECK(cudaFree(d_ones_xn));
    CUDA_CHECK(cudaFree(d_QLam));
    CUDA_CHECK(cudaFree(d_Qt_ones));

    // Orthogonalize U
    double* d_UU;
    CUDA_CHECK(cudaMalloc((void**)&d_UU, x_n * k * sizeof(double)));
    orthogonalize_cuda(cublasH, cusolverH, d_U, d_Lam, x_n, k, d_UU);

    // Copy to arma::mat d_UU
    arma::mat UU(x_n, k);
    CUDA_CHECK(cudaMemcpy(UU.memptr(), d_UU, x_n * k * sizeof(double), cudaMemcpyDeviceToHost));
    // Cluster the approximated eigenvectors
    arma::mat centroids;
    arma::uvec y_hat(x_n);  
    bool status = arma::kmeans(centroids, UU.t(), k, arma::random_subset, 30, false);
    if (!status) {
        std::cout << "Clustering failed!" << std::endl;
        this->y_hat = y_hat;
    }
    centroids = centroids.t();
    arma::vec d(k);
    for (unsigned int i = 0; i < x_n; i++) {
        for (unsigned int j = 0; j < k; j++) {
            d.row(j) = arma::norm(UU.row(i) - centroids.row(j));
        }
        y_hat.row(i) = d.index_min();
    }

    this->y_hat = y_hat;

    // Free unneded memory: d_U, d_UU, d_Lam
    CUDA_CHECK(cudaFree(d_Lam));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_UU));
    stopTime(&tim);
    printElapsedTime(tim, "GPU", GREEN);
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
}

void cpu_affinity_matrix(arma::mat &A_11, arma::mat &Z)
{
    int m = Z.n_rows;
    double mu = 0.0;
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < m; j++) {
            mu += pow(arma::norm(Z.row(i) - Z.row(j)), 2);
        }
    }
    mu /= pow(m, 2);
    mu = 1 / mu;
    printf("mu: %f\n", mu);

    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=i; j<m; j++) {
            double val = exp(-mu * pow(arma::norm(Z.row(i) - Z.row(j)), 2));
            A_11(i, j) = val;
            A_11(j, i) = val;
        }
    }
}

void cpu_m_star(arma::mat &M_star, arma::mat &A_11)
{
    int m = A_11.n_rows;
    arma::vec ww = A_11 * arma::ones<arma::vec>(m);
    arma::mat D_star_ = arma::diagmat(arma::pow(arma::sqrt(ww), -1));
    M_star = D_star_ * A_11 * D_star_;
}

void CSSC::test()
{
    // Initialize cublas
    cublasHandle_t handle;

    int n = X.n_rows;
    // Sample m points from X (Armadillo)
    arma::uvec inds = arma::linspace<arma::uvec>(0, n-1, n);
    inds = arma::shuffle(inds);
    inds = inds.rows(0, m-1);
    arma::mat Z = X.rows(inds);

    // Calculate Gaussian kernel
    Timer timer;
    startTime(&timer);
    arma::mat A_11(m, m);
    cpu_affinity_matrix(A_11, Z);

    arma::mat M_star;
    cpu_m_star(M_star, A_11);
    arma::vec eigval;
    arma::mat eigvec;
    arma::eig_sym(eigval, eigvec, M_star);
    stopTime(&timer);
    printElapsedTime(timer, "CPU execution", CYAN);
    A_11.save("A_11.csv", arma::csv_ascii);
    M_star.save("M_star.csv", arma::csv_ascii);


    // // Calculate the affinity matrix (CUDA)
    arma::mat A_11_cu(m, m);
    arma::mat M_star_cu(m, m);
    double mu = 1.2734e7; // Not important anymore
    cublasCreate(&handle);
    test_calculate_affinity_matrix(A_11_cu, Z, mu);
    test_calculate_m_star(handle, A_11_cu, M_star_cu);
    test_eig_kernel(M_star_cu, m);

    A_11_cu.save("A_11_cu.csv", arma::csv_ascii);
    M_star_cu.save("M_star_cu.csv", arma::csv_ascii);

    // Test if A_11 and A_11_cu are equal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j<m; j++) {
            if (A_11(i, j) != A_11_cu(i, j)) {
                cout << "A_11 and A_11_cu are not equal at: " << i << "," << j << endl;
                cout << "A_11: " << A_11(i, j) << endl;
                cout << "A_11_cu: " << A_11_cu(i, j) << endl;
                return;
            }
        }
    }
    cout << "A_11 and A_11_cu are equal!" << endl;

    // Test if M_star and M_star_cu are equal
    for (unsigned int i = 0; i < m; i++) {
        for (unsigned int j = 0; j < m; j++) {
            if (M_star(i, j) != M_star_cu(i, j)) {
                cout << "M_star and M_star_cu are not equal at: " << i << "," << j << endl;
                cout << "M_star: " << M_star(i, j) << endl;
                cout << "M_star_cu: " << M_star_cu(i, j) << endl;
                return;
            }
        }
    }
    cout << "M_star and M_star_cu are equal!" << endl;
    
    cublasDestroy(handle);
}