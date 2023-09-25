#ifndef FIND_OPTIMAL_K_HPP
#define FIND_OPTIMAL_K_HPP

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

int find_optimal_k(arma::mat& X, int k, arma::vec& eigvals);

#endif