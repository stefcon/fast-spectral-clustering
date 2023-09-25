#ifndef SAMPLE_H
#define SAMPLE_H

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <vector>
#include <string>

void read_labels(std::string& file_name, arma::uvec& uY);
std::vector<int> sample_without_replacement(int low, int high, int n);

#endif