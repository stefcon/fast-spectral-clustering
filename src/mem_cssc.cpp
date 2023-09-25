#include "../lib/mem_cssc.hpp"
#include "../lib/cuda_helper.h"
#include "../lib/timer.h"
#include "../lib/kernels.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>


void MemCSSC::read_data_dimensions(std::string& filepath)
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
            n = std::count(line.begin(), line.end(), ',');+ 1; // TODO: fix it afterwards, needs to be done because of Y currently
        }
    }
    infile.close();
}

void MemCSSC::sample_matrix_X(arma::mat& Z, double *d_Z, int n)
{
    arma::uvec inds_chosen = arma::conv_to<arma::uvec>::from(sample_without_replacement(0, x_n-1, m));

    // arma::uvec inds_chosen;
    // inds_chosen.load("inds_chosen.csv", arma::csv_ascii);
    // inds_for_sampling = arma::shuffle(inds_for_sampling);
    std::sort(inds_for_sampling.begin(), inds_for_sampling.end());
    arma::uvec inds_arma = inds_for_sampling(inds_chosen);

    std::sort(inds_arma.begin(), inds_arma.end());
    std::vector<int> inds = arma::conv_to<std::vector<int>>::from(inds_arma);
    


    // inds_chosen.save("inds_chosen.csv", arma::csv_ascii);
    // inds_arma.save("inds_arma.csv", arma::csv_ascii);
    // inds_for_sampling.save("inds_for_sampling.csv", arma::csv_ascii);
    
    std::ifstream infile(filepath);
    std::string line;
    int i = 0;
    int curr_row = 0;
    while (std::getline(infile, line))
    {
        // if (std::find(inds.begin(), inds.end(), i) != inds.end())
        if (std::binary_search(inds.begin(), inds.end(), i))
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

void MemCSSC::calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH)
{
    Timer tim;
    startTime(&tim);

    arma::mat Q(x_n, k);
    double* d_a, *d_q;
    arma::mat onesies = arma::ones(1, m);
    // double* d_Z;
    // CUDA_CHECK(cudaMalloc((void**)&d_Z, m * n * sizeof(double)));
    // CUDA_CHECK(cudaMemcpy(d_Z, Z.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice));

    // Transpoee matrices for better locality
    Z = Z.t();
    Q = Q.t();
    B = B.t();
    // omp_set_nested(1);
    #pragma omp parallel shared(Q, Z, B) private(d_a, d_q)
    {
        std::ifstream infile(filepath);
        std::string line;
        int line_num = 0;
    
        // CUDA_CHECK(cudaMalloc((void**)&d_a, m * sizeof(double)));
        // CUDA_CHECK(cudaMalloc((void**)&d_q, k * sizeof(double)));
        // double alpha = 1.0;
        // double beta = 0.0;

        // double* d_x, *d_result;
        // CUDA_CHECK(cudaMalloc((void**)&d_x, n * sizeof(double)));
        // CUDA_CHECK(cudaMalloc((void**)&d_result, m * n * sizeof(double)));
        

        #pragma omp for
        for (unsigned int i = 0; i < x_n; i++) {
            // Read line inds_for_sampling[i] from dataset file
            arma::vec x(n);
            int next_ind = inds_for_sampling[i];
            while (std::getline(infile, line))
            {
                if (line_num == next_ind)
                {
                    // Option 1
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
            arma::vec a(m);
            #pragma unroll
            for (unsigned int j = 0; j < m; j++)
            {
                // a(j) = arma::norm(x - Z.col(j));
                a(j) = arma::norm(x - Z.col(j));
            }
            Q.col(i) = std::move(B * a);

            // Also a lot slower!
            // Precompute differences if x and Z are not changing frequently
            // arma::mat diffMatrix = x * onesies - Z; // Broadcasting x and subtracting from each column of Z

            // // Compute norms for all columns
            // arma::vec a = std::move(arma::sqrt(arma::sum(diffMatrix % diffMatrix, 0)).t()); // Element-wise square, sum, and square root

            // // Update Q
            // Q.col(i) = B * a;


            // Not fast enough because of redundant memory transfers
            // --------------------------------------------------------------------------------------------
            // CUDA_CHECK(cudaMemcpy(d_x, x.memptr(), n * sizeof(double), cudaMemcpyHostToDevice));
            // calculate_affinity_row_cuda(cublasH, d_Z, d_a, d_x, d_result, m, n);
            // --------------------------------------------------------------------------------------------
            
            // Copy a to GPU            
            // CUDA_CHECK(cudaMemcpy(d_a, a.memptr(), m * sizeof(double), cudaMemcpyHostToDevice));
            // CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_T, m, k, &alpha, d_B, m, d_a, 1, &beta, d_q, 1));
            // CUBLAS_CHECK(cublasDcopy(cublasH, k, d_q, 1, d_Q + i, x_n));
            
        }

        infile.close();
        // CUDA_CHECK(cudaFree(d_a));
        // CUDA_CHECK(cudaFree(d_q));
        // CUDA_CHECK(cudaFree(d_x));
        // CUDA_CHECK(cudaFree(d_result));
    }
    // Transpose Q back to original dimensions
    Q = Q.t();
    // Z = Z.t();
    B = B.t();
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
    stopTime(&tim);
    printElapsedTime(tim, "calculate_affinity_Q", DGREEN);
}

// --------------------------------------------------------------------------------------------

void SparseMemCSSC::read_data_dimensions(std::string& filepath)
{
    x_n = 0;
    n = 0;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line))
    {
        x_n++;
        // Iterate through the line and see which j is the largest
        std::istringstream iss(line);
        std::string token;
        while (std::getline(iss, token, ','))
        {
            // Turn number before ':' into int
            int curr_n = std::stoi(token.substr(0, token.find(':')));
            if (curr_n + 1 > n) n = curr_n + 1;
        }
    }
    std::cout << "x_n: " << x_n << std::endl;
    std::cout << "n: " << n << std::endl;
}


void SparseMemCSSC::sample_matrix_X(arma::mat& Z, double *d_Z, int n)
{
    arma::uvec inds_chosen = arma::conv_to<arma::uvec>::from(sample_without_replacement(0, x_n-1, m));

    std::sort(inds_for_sampling.begin(), inds_for_sampling.end());
    arma::uvec inds_arma = inds_for_sampling(inds_chosen);

    std::sort(inds_arma.begin(), inds_arma.end());
    std::vector<int> inds = arma::conv_to<std::vector<int>>::from(inds_arma);
    
    
    // Initialize Z to zeros
    Z.zeros();

    std::ifstream infile(filepath);
    std::string line;
    int i = 0;
    int curr_row = 0;
    while (std::getline(infile, line))
    {
        // if (std::find(inds.begin(), inds.end(), i) != inds.end())
        if (std::binary_search(inds.begin(), inds.end(), i))
        {
            if (line.empty()) {
                curr_row++;
                continue;   
            }
                
            std::istringstream iss(line);
            std::string token;
            int j;
            double val;
            char delim;
            while (std::getline(iss, token, ','))
            {
                std::istringstream iss2(token);
                iss2 >> j >> delim >> val;
                Z(curr_row, j) = val;
            }
            curr_row++;
        }
        i++;
    }
    infile.close();
    CUDA_CHECK(cudaMemcpy(d_Z, Z.memptr(), m * n * sizeof(double), cudaMemcpyHostToDevice));
}

void SparseMemCSSC::calculate_affinity_Q(arma::mat& Z, arma::mat& B, int n, double* d_Q, double* d_B, cublasHandle_t cublasH)
{
    Timer tim;
    startTime(&tim);

    arma::mat Q(x_n, k);

    // Transpoee matrices for better locality
    Z = Z.t();
    Q = Q.t();
    B = B.t();
    // omp_set_nested(1);
    #pragma omp parallel shared(Q, Z, B)
    {
        std::ifstream infile(filepath);
        std::string line;
        int line_num = 0;
        
        #pragma omp for
        for (unsigned int i = 0; i < x_n; i++) {
            // Read line inds_for_sampling[i] from dataset file
            arma::vec x(n); x.zeros();
            int next_ind = inds_for_sampling[i];
            while (std::getline(infile, line))
            {
                if (line_num == next_ind)
                {
                    std::istringstream iss(line);
                    std::string token;
                    int j;
                    double val;
                    char delim;
                    while (std::getline(iss, token, ','))
                    {
                        std::istringstream iss2(token);
                        iss2 >> j >> delim >> val;
                        x(j) = val;
                    }
                    break;
                }
                line_num++;
            }
            // Prepare the cursor for the next iteration
            line_num++;
            
            // Calculate the i-th row of Q
            arma::vec a(m);
            #pragma unroll
            for (unsigned int j = 0; j < m; j++)
            {
                a(j) = arma::norm(x - Z.col(j));
            }
            Q.col(i) = std::move(B * a);
        }

        infile.close();
    }
    // Transpose Q back to original dimensions
    Q = Q.t();
    Z = Z.t();
    B = B.t();
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
    stopTime(&tim);
    printElapsedTime(tim, "calculate_affinity_Q", DGREEN);
}