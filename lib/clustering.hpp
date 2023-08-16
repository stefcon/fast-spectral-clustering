#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

// Base class for all clustering algorithms
// TODO: Possibly add some attributes for storing the results and parameters
class Clustering {
public:
    Clustering() {}

    virtual void fit() = 0;
    double accuracy(const arma::uvec& Y, const arma::uvec &y_hat);

    virtual ~Clustering() {}
};

#endif // CLUSTERING_HPP