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
    for (unsigned int i = 0; i < m; i++) {
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
        for (unsigned int j = 0; j < m; j++)
        {
            a.col(j) = arma::norm(X.row(i) - Z.row(j));
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
    Timer tim_ort;
    startTime(&tim_ort);
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
    stopTime(&tim_ort);
    printElapsedTime(tim_ort, "CPU orthogonalization", CYAN);

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

// Protected methods (needed for previous "gpu_fit" method, non-existant in the current version)
//// -----------------------------------------------------------------------------------------------
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


void CSSC::calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH)
{
    arma::mat Q(x_n, k);
    Z = Z.t();
    Q = Q.t();
    B = B.t();
    X = X.t();
    #pragma omp parallel for
    for (unsigned int i = 0; i < x_n; i++) {
        // arma::rowvec a(m);
        // arma::rowvec x = X.row(i);
        arma::vec a(m);
        arma::vec x = X.col(i);
        for (unsigned int j = 0; j < m; j++)
        {
            // a.col(j) = arma::norm(x - Z.row(j));
            a(j) = arma::norm(x - Z.col(j));
        }
        // Q.row(i) = a * B;
        Q.col(i) = B * a;
    }
    Z = Z.t();
    Q = Q.t();
    B = B.t();
    X = X.t();
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
}
//// -----------------------------------------------------------------------------------------------