#ifndef CSSC_HPP
#define CSSC_HPP

#include "clustering.hpp"
#include "sample.hpp"
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cublas_v2.h>
#include <cusolverDn.h>

class CSSC : public Clustering {
public:
    CSSC(int k, int m) : k(k), m(m) {};
    CSSC(arma::mat X_arg, int k, int m) : X(X_arg), k(k), m(m) 
    {
        x_n = X.n_rows;
        n = X.n_cols;
    }
    ~CSSC() {};

    void fit() override;
    void gpu_fit(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH);

    arma::uvec get_y_hat() const { return y_hat; }

    void test();
protected:
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix A_11 (expecting column-major matrices)
    // A_11 - output, Z - input, m - number of rows, n - number of columns
    virtual void calculate_affinity_matrix_A(double* d_A_11, double* d_Z, int m, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q);

    int k, m, x_n, n;
    arma::uvec y_hat;
private:
    arma::mat X;
};

#endif // CSSC_HPP