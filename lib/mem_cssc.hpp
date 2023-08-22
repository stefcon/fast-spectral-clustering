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
    }
    ~MemCSSC() {};

protected:
    // Read number of rows and columns in data matrix
    virtual void read_data_dimensions(std::string filepath);
    // Sample matrix X from 
    virtual void sample_matrix_X(arma::mat& Z, double *d_Z, int n);
    // Affinity matrix using between points x_i and points in Z matrix
    virtual void calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q);

    std::string filepath;
};


#endif // MEM_CSSC_HPP