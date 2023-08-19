#ifndef SAMPLE_H
#define SAMPLE_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>

arma::uvec sample_without_replacement(int low, int high, int n);


#endif