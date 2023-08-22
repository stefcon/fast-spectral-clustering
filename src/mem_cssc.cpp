#include "../lib/mem_cssc.hpp"
#include "../lib/cuda_helper.h"
#include <fstream>
#include <algorithm>


void MemCSSC::read_data_dimensions(std::string filepath)
{
    x_n = 0;
    n = 0;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line))
    {
        x_n++;
        if (n == 0)
        {
            n = std::count(line.begin(), line.end(), ',') + 1; // TODO: fix it afterwards, needs to be done because of Y currently
        }
    }
    infile.close();
}

void MemCSSC::sample_matrix_X(arma::mat& Z, double *d_Z, int n)
{
    std::vector<int> inds = sample_without_replacement(0, x_n - 1, m);
    arma::uvec inds_arma = arma::conv_to<arma::uvec>::from(inds);
    
    std::ifstream infile(filepath);
    std::string line;
    int i = 0;
    int curr_row = 0;
    while (std::getline(infile, line))
    {
        if (std::find(inds.begin(), inds.end(), i) != inds.end())
        {
            std::istringstream iss(line);
            std::string token;
            int j = 0;
            while (std::getline(iss, token, ','))
            {
                Z(curr_row, j) = std::stod(token);
                j++;
                if (j == n) break;
            }
            curr_row++;
        }
        i++;
    }
    infile.close();
    CUDA_CHECK(cudaMemcpy(d_Z, Z.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice));
}

void MemCSSC::calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q)
{
    arma::mat Q(x_n, k);
    #pragma omp parallel shared(Q, Z, B)
    {
        std::ifstream infile(filepath);
        std::string line;
        int line_num = 0;
        #pragma omp for private(line)
        for (unsigned int i = 0; i < x_n; i++) {
            // Read line i from dataset file
            arma::rowvec x(n);
            while (std::getline(infile, line))
            {
                if (line_num == i)
                {
                    std::istringstream iss(line);
                    std::string token;
                    int j = 0;
                    while (std::getline(iss, token, ','))
                    {
                        x(j) = std::stod(token);
                        j++;
                        if (j == n) break;
                    }
                    break;
                }
                line_num++;
            }
            // Prepare the cursor for the next iteration
            line_num++;

            // Calculate the i-th row of Q
            arma::rowvec a(m);
            for (unsigned int j = 0; j < m; j++)
            {
                a.col(j) = arma::norm(x - Z.row(j));
            }
            Q.row(i) = a * B;
        }
        infile.close();
    }
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
}