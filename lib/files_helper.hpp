#ifndef FILES_HELPER_HPP
#define FILES_HELPER_HPP

#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <string>

void get_data_dimensions_csv(std::string& filepath, int& m, int& n);
void get_data_dimensions_coo(std::string& filepath, int& m, int& n);

void read_matrix_X_sparse_coo(std::string& filepath, arma::mat& X);


#endif