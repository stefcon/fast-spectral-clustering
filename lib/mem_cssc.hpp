#ifndef MEM_CSSC_HPP
#define MEM_CSSC_HPP

#include "cssc.hpp"
#include <string>

class MemCSSC : public Clustering 
{
public:
    MemCSSC(std::string filepath, int x_n, int n, int k, int m) : filepath(filepath), x_n(x_n), n(n), k(k), m(m)
    {
        // Initialize inds_for_sampling to all indices
        inds_for_sampling = arma::linspace<arma::uvec>(0, x_n-1, x_n);
    }

    MemCSSC(std::string& filepath, arma::uvec inds, int n, int k, int m) : filepath(filepath), inds_for_sampling(inds), n(n), k(k), m(m)
    {
        x_n = inds.n_elem;
    }

    void fit() override;

    ~MemCSSC() {};

protected:
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix A_11 (expecting column-major matrices)
    // A_11 - output, Z - input, m - number of rows, n - number of columns
    virtual void calculate_affinity_matrix_A(double* d_A_11, double* d_Z, int m, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH);


    std::string filepath;
    arma::uvec inds_for_sampling;
    int k, m, x_n, n;
};

class SparseMemCSSC : public MemCSSC
{
public:
    SparseMemCSSC(std::string filepath, int x_n, int n, int k, int m) : MemCSSC(filepath, x_n, n, k, m)
    {
        // Initialize inds_for_sampling to all indices
        inds_for_sampling = arma::linspace<arma::uvec>(0, x_n-1, x_n);
    }

    SparseMemCSSC(std::string& filepath, arma::uvec inds, int n, int k, int m) : MemCSSC(filepath, inds, n, k, m)
    {
        x_n = inds.n_elem;
    }

    ~SparseMemCSSC() {};

protected:
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH);
};


#endif // MEM_CSSC_HPP