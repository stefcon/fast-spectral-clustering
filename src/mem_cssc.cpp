#include "../lib/mem_cssc.hpp"
#include "../lib/cuda_helper.h"
#include "../lib/timer.h"
#include "../lib/kernels.hpp"
#include <fstream>
#include <algorithm>
#include <iostream>


void MemCSSC::fit()
{
    cublasHandle_t cublasH;
    cublasCreate(&cublasH);
    // Initialize cusolver
    cusolverDnHandle_t cusolverH;
    cusolverDnCreate(&cusolverH);
    // Cublas constants (used in cublas calsl)
    const double alpha = 1.0;
    const double beta = 0.0;

    Timer tim;
    startTime(&tim);

    // Calculate affinity matrix
    double* d_A_11;
    double* d_Z;
    CUDA_CHECK(cudaMalloc((void**)&d_A_11, m * m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Z, m * n * sizeof(double)));
    // Sample m matrix rows from X and copy to d_Z
    arma::mat Z(m, n);
    sample_matrix_X(Z, d_Z, n);
    calculate_affinity_matrix_A(d_A_11, d_Z, m, n);
    // Free unneded memory: d_Z (still on CPU)
    CUDA_CHECK(cudaFree(d_Z));

    // Calculate M_star
    double* d_M_star;
    double* d_ww;
    double* d_ones;
    CUDA_CHECK(cudaMalloc((void**) &d_ww, m * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**) &d_ones, m * sizeof(double)));

    // Initialize d_ones_m
    ones_cuda(d_ones, m);
    // ww = A_11 * ones_m
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, m, m, &alpha, d_A_11, m, d_ones, 1, &beta, d_ww, 1));

    // M_star = D_star_ * A_11 * D_star_
    pow_vec(d_ww, m, -0.5);
    gemv_diag(d_A_11, d_ww, m, m, MUL_LEFT_T);
    gemv_diag(d_A_11, d_ww, m, m, MUL_RIGHT_T);
    // A_11 now holds M_star (overwritten)
    d_M_star = d_A_11;

    // Free unneded memory: d_ones
    CUDA_CHECK(cudaFree(d_ones));


    // Find the eigendecomp of M_star
    double* d_eigvals;
    double* d_eigvecs;
    double* d_W;
    CUDA_CHECK(cudaMalloc((void**)&d_eigvals, k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_eigvecs, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, sizeof(double) * m));
    eig_dsymx_cusolver(cusolverH, d_M_star, d_W, m, k, d_eigvals, d_eigvecs);
    // d_M_star has been overwritten by eigenvectors, so we 
    // can deallocoate it as well as d_W
    CUDA_CHECK(cudaFree(d_M_star));
    CUDA_CHECK(cudaFree(d_W));

    // Calculate B
    double* d_B;
    double* d_Lam;
    CUDA_CHECK(cudaMalloc((void**)&d_B, m * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Lam, k * k * sizeof(double)));

    // Diagonalize eigvals
    diagmat_cublas(cublasH, d_eigvals, d_Lam, k);
    
    // B = D_star_ (ww ^-1/2 diagonalized) * eigvecs * pow(eigvals, -1)
    gemv_diag(d_eigvecs, d_ww, m, k, MUL_LEFT_T);
    pow_vec(d_eigvals, k, -1);
    gemv_diag(d_eigvecs, d_eigvals, m, k, MUL_RIGHT_T);
    d_B = d_eigvecs;
    // Free unneded memory: d_eigvals
    CUDA_CHECK(cudaFree(d_eigvals));

    // Copy B to host arma::mat
    arma::mat B(m, k);
    CUDA_CHECK(cudaMemcpy(B.memptr(), d_B, m * k * sizeof(double), cudaMemcpyDeviceToHost));

    // Calcualte Q on cpu 
    double* d_Q;
    CUDA_CHECK(cudaMalloc((void**)&d_Q, x_n * k * sizeof(double)));
    calculate_affinity_Q(Z, B, n, d_Q, d_B, cublasH);
    // Moved while testing for better performance!
    CUDA_CHECK(cudaFree(d_B));

    double* d_ones_xn;
    double* d_dd;
    double* d_QLam;
    double* d_Qt_ones;
    double* d_U;
    CUDA_CHECK(cudaMalloc((void**)&d_ones_xn, x_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_dd, x_n * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_QLam, x_n * k * sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&d_Qt_ones, k * sizeof(double)));

    // Initialize d_ones_nx
    ones_cuda(d_ones_xn, x_n);
    // dd = Q * Lam * Q.t() * d_ones_nx
    CUBLAS_CHECK(cublasDgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, x_n, k, k, &alpha, d_Q, x_n, d_Lam, k, &beta, d_QLam, x_n));

    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_T, x_n, k, &alpha, d_Q, x_n, d_ones_xn, 1, &beta, d_Qt_ones, 1));
    
    CUBLAS_CHECK(cublasDgemv(cublasH, CUBLAS_OP_N, x_n, k, &alpha, d_QLam, x_n, d_Qt_ones, 1, &beta, d_dd, 1));
    

    // U = diagmat(pow(dd, -0.5)) * Q
    pow_vec(d_dd, x_n, -0.5);
    gemv_diag(d_Q, d_dd, x_n, k, MUL_LEFT_T);
    // Q now holds U (overwritten)
    d_U = d_Q;
    
    // Free unneded memory: d_dd, d_ones_xn, d_QLam, d_Qt_ones
    CUDA_CHECK(cudaFree(d_dd));
    CUDA_CHECK(cudaFree(d_ones_xn));
    CUDA_CHECK(cudaFree(d_QLam));
    CUDA_CHECK(cudaFree(d_Qt_ones));

    // Orthogonalize U
    double* d_UU;
    CUDA_CHECK(cudaMalloc((void**)&d_UU, x_n * k * sizeof(double)));
    orthogonalize_cuda(cublasH, cusolverH, d_U, d_Lam, x_n, k, d_UU);

    // Copy to arma::mat d_UU
    arma::mat UU(x_n, k);
    CUDA_CHECK(cudaMemcpy(UU.memptr(), d_UU, x_n * k * sizeof(double), cudaMemcpyDeviceToHost));
    // Cluster the approximated eigenvectors
    arma::mat centroids;
    arma::uvec y_hat(x_n);  
    bool status = arma::kmeans(centroids, UU.t(), k, arma::random_subset, 30, false);
    if (!status) {
        std::cout << "Clustering failed!" << std::endl;
        this->y_hat = y_hat;
    }
    centroids = centroids.t();
    arma::vec d(k);
    for (unsigned int i = 0; i < x_n; i++) {
        for (unsigned int j = 0; j < k; j++) {
            d.row(j) = arma::norm(UU.row(i) - centroids.row(j));
        }
        y_hat.row(i) = d.index_min();
    }

    this->y_hat = y_hat;

    // Free unneded memory: d_U, d_UU, d_Lam
    CUDA_CHECK(cudaFree(d_Lam));
    CUDA_CHECK(cudaFree(d_U));
    CUDA_CHECK(cudaFree(d_UU));
    stopTime(&tim);
    printElapsedTime(tim, "GPU", GREEN);
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);
}


// void MemCSSC::read_data_dimensions(std::string& filepath)
// {
//     x_n = 0;
//     n = 0;
//     std::ifstream infile(filepath);
//     std::string line;
//     while (std::getline(infile, line))
//     {
//         x_n++;
//         if (n == 0)
//         {
//             n = std::count(line.begin(), line.end(), ',');+ 1; // TODO: fix it afterwards, needs to be done because of Y currently
//         }
//     }
//     infile.close();
// }

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

void MemCSSC::calculate_affinity_matrix_A(double* d_A_11, double* d_Z, int m, int n)
{
    calculate_affinity_matrix_cuda(d_A_11, d_Z, m, n);
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

            // --------------------------------------------------------------------------------------------
            // A lot slower than above implementation!
            // --------------------------------------------------------------------------------------------
            // Precompute differences if x and Z are not changing frequently
            // arma::mat diffMatrix = x * onesies - Z; // Broadcasting x and subtracting from each column of Z

            // // Compute norms for all columns
            // arma::vec a = std::move(arma::sqrt(arma::sum(diffMatrix % diffMatrix, 0)).t()); // Element-wise square, sum, and square root

            // // Update Q
            // Q.col(i) = B * a;
            // --------------------------------------------------------------------------------------------
            // --------------------------------------------------------------------------------------------
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
    // Transpose matrices back to original dimensions
    Q = Q.t();
    Z = Z.t();
    B = B.t();
    // Copy Q to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, Q.memptr(), x_n * k * sizeof(double), cudaMemcpyHostToDevice));
    stopTime(&tim);
    printElapsedTime(tim, "calculate_affinity_Q", DGREEN);
}

// --------------------------------------------------------------------------------------------

// void SparseMemCSSC::read_data_dimensions(std::string& filepath)
// {
//     x_n = 0;
//     n = 0;
//     std::ifstream infile(filepath);
//     std::string line;
//     while (std::getline(infile, line))
//     {
//         x_n++;
//         // Iterate through the line and see which j is the largest
//         std::istringstream iss(line);
//         std::string token;
//         while (std::getline(iss, token, ','))
//         {
//             // Turn number before ':' into int
//             int curr_n = std::stoi(token.substr(0, token.find(':')));
//             if (curr_n + 1 > n) n = curr_n + 1;
//         }
//     }
//     std::cout << "x_n: " << x_n << std::endl;
//     std::cout << "n: " << n << std::endl;
// }


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