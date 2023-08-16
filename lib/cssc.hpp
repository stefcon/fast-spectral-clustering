#ifndef CSSC_HPP
#define CSSC_HPP

#include "clustering.hpp"
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cublas_v2.h>
#include <cusolverDn.h>

class CSSC : public Clustering {
public:
    CSSC(arma::mat X, int k, int m) : X(X), k(k), m(m) 
    {
        inds = arma::linspace<arma::uvec>(0, X.n_rows - 1, X.n_rows);
        inds = arma::shuffle(inds);
    }
    ~CSSC() {};

    void fit() override;
    void gpu_fit(cublasHandle_t cublasH, cusolverDnHandle_t cusolverH);
    arma::uvec get_y_hat() const { return y_hat; }

    void test();
private:
    int k, m;
    arma::mat X;
    arma::uvec y_hat;
    arma::uvec inds;
};

#endif