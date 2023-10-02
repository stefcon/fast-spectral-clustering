#ifndef CLUSTERING_HPP
#define CLUSTERING_HPP

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

// Base class for all clustering algorithms
class Clustering {
public:
    Clustering() {}

    virtual void fit() = 0;

    virtual arma::uvec get_y_hat() const { return y_hat;}
    double accuracy(const arma::uvec& Y, const arma::uvec &y_hat);

    virtual ~Clustering() {}
protected:
    arma::uvec y_hat;
};

#endif // CLUSTERING_HPP