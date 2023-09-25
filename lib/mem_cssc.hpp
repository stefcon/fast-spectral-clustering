#ifndef MEM_CSSC_HPP
#define MEM_CSSC_HPP

#include "cssc.hpp"
#include <string>

class MemCSSC : public CSSC 
{
public:
    MemCSSC(std::string filepath, int k, int m) : CSSC(k, m), filepath(filepath)
    {
        read_data_dimensions(filepath);
        // Initialize inds_for_sampling to all indices
        inds_for_sampling = arma::linspace<arma::uvec>(0, x_n-1, x_n);
    }

    MemCSSC(std::string& filepath, arma::uvec inds, int k, int m) : CSSC(k, m), filepath(filepath), inds_for_sampling(inds)
    {
        read_data_dimensions(filepath);
        x_n = inds.n_elem;
    }

    ~MemCSSC() {};

protected:
    // Read number of rows and columns in data matrix
    virtual void read_data_dimensions(std::string& filepath);
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH);

    std::string filepath;
    arma::uvec inds_for_sampling;
};

class SparseMemCSSC : public CSSC
{
public:
    SparseMemCSSC(std::string filepath, int k, int m) : CSSC(k, m), filepath(filepath)
    {
        read_data_dimensions(filepath);
        // Initialize inds_for_sampling to all indices
        inds_for_sampling = arma::linspace<arma::uvec>(0, x_n-1, x_n);
    }

    SparseMemCSSC(std::string& filepath, arma::uvec inds, int k, int m) : CSSC(k, m), filepath(filepath), inds_for_sampling(inds)
    {
        read_data_dimensions(filepath);
        x_n = inds.n_elem;
    }

    ~SparseMemCSSC() {};

protected:
    // Read number of rows and columns in data matrix
    virtual void read_data_dimensions(std::string& filepath);
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH);

    std::string filepath;
    arma::uvec inds_for_sampling;
};


#endif // MEM_CSSC_HPP