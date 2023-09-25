#include "../lib/cssc.hpp"
#include "../lib/mem_cssc.hpp"
#include "../lib/sample.hpp"
#include "../lib/timer.h"
#include "../lib/files_helper.hpp"
#include "../lib/optimal_k.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
#include <queue>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>


using namespace std;
namespace fs = std::filesystem;

const int MNIST8M_SIZE = 8100000;
const int MNIST8M_SUBSET_SIZE = 4130460;
const int MNIST8M_DIM = 784;

const int NUM_ITER = 10;

void tester(Clustering& clustering, arma::uvec uY, string label_folder_name)
{
    vector<double> accuracies;
    vector<arma::uvec> labels;
    double accur;
    arma::uvec y_hat;
    for (int iter = 0; iter < NUM_ITER; ++iter) {
        cout << "Iteration: " << iter << endl;
        clustering.gpu_fit();
        y_hat = clustering.get_y_hat();
        
        accur = clustering.accuracy(uY, clustering.get_y_hat());
        cout << "Accuracy: " << accur << endl;
        accuracies.push_back(accur);
        labels.push_back(y_hat);
    }
    int sz = accuracies.size();
    double mean = std::reduce(accuracies.begin(), accuracies.end()) / sz;
    auto variance_func = [&mean, &sz](double accumulator, const double& val) {
        return accumulator + ((val - mean)*(val - mean) / (sz - 1));
    };
    double variance = std::accumulate(accuracies.begin(), accuracies.end(), 0.0, variance_func);

    cout << mean << " +- " << variance << endl;

    // Store all labels in seperate files, in a new folder
    string folder_name = "data/processed/" + label_folder_name;
    fs::create_directory(folder_name);
    for (int i = 0; i < labels.size(); ++i) {
        string file_name = folder_name + "/iter" + to_string(i) + ".csv";
        labels[i].save(file_name, arma::csv_ascii);
    }
}

void runner(Clustering& clustering, string label_folder_name, int num_iter = NUM_ITER)
{
    vector<arma::uvec> labels;
    arma::uvec y_hat;
    string folder_name = "data/processed/" + label_folder_name;
    fs::create_directories(folder_name);
    for (int iter = 0; iter < num_iter; ++iter) {
        clustering.gpu_fit();

        y_hat = clustering.get_y_hat();
        labels.push_back(y_hat);
    }

    // Store all labels in seperate files, in a new folder
    for (int i = 0; i < labels.size(); ++i) {
        string file_name = folder_name + "/iter" + to_string(i) + ".csv";
        labels[i].save(file_name, arma::csv_ascii);
    }

}

void transform_file(string source_file, string target_file, int last_label) {
    fstream source(source_file, ios::in);
    fstream target(target_file, ios::out);

    string line;
    int label;
    while (getline(source, line)) {
        label = stoi(line.substr(line.find_last_of(',') + 1));
        if (label <= last_label) {
            target << line << endl;
        }
    }
}

void test_coo(string& file_name) 
{
    arma::mat Z(3, 21);
    std::vector<int> inds{0, 2, 4};
    std::ifstream infile(file_name);
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
                cout << j << ", " << val << endl;
                Z(curr_row, j) = val;
            }
            curr_row++;
        }
        i++;
    }
    infile.close();

    Z.save("Z_test.csv", arma::csv_ascii);
}

vector<int> get_top_k_differences(arma::vec& vec, int k)
{
    priority_queue<pair<double, int>> pq;
    arma::vec diffs = arma::abs(arma::diff(vec));
    for (int i = 0; i < diffs.n_elem; ++i) {
        pq.push(make_pair(diffs(i), i));
    }
    vector<int> inds;
    for (int i = 0; i < k; ++i) {
        cout << pq.top().first << " " << pq.top().second << endl;
        inds.push_back(pq.top().second + 1);
        pq.pop();
    }
    return inds;
}

// int main(int argc, char* argv[])
// {
//     cudaDeviceSynchronize();

//     // string file_name = "data/processed/usps.csv";
//     string file_name = "data/processed/Data_C5.npzcoo.csv";

//     int m, n;
//     arma::arma_rng::set_seed_random();

    // arma::vec eigvals;
    // eigvals.load("eigvals.csv", arma::csv_ascii);
    // auto inds = get_top_k_differences(eigvals, 15);
    // for (auto ind : inds) {
    //     cout << ind << ", ";
    // }

//     // int k = 50;

//     // get_data_dimensions(file_name, m, n);
//     // arma::mat X(m,n);
//     // read_matrix_X_sparse(file_name, X);
//     // // ---------------------------
//     // // arma::mat X;
//     // // X.load(file_name, arma::csv_ascii);
//     // // X = X.cols(0, X.n_cols-2);
//     // // std::cout << "X: " << X.n_rows << ", " << X.n_cols << std::endl;
    
//     // arma::vec eigvals(k);
//     // int result = find_optimal_k(X, k, eigvals);
//     // cout << "Optimal k: " << result << endl;

//     return 0;
// }

// int main(int argc, char* argv[])
// {   

//     cudaDeviceSynchronize();
//     cublasHandle_t cublasH;
//     cusolverDnHandle_t cusolverH;
//     cublasCreate(&cublasH);
//     cusolverDnCreate(&cusolverH);

//     // string file_name = "data/processed/mnist.csv";
//     string file_name = "data/processed/Data_C5.npzcoo.csv";
//     arma::arma_rng::set_seed_random();
    
//     int m = 1000;
//     for (int k = 2; k <= 9; k++) {
//         SparseMemCSSC mem_cssc_clustering(file_name, k, m);
//         runner(mem_cssc_clustering, "mem_cssc_bgi/" + std::to_string(k), 4);
//     }
    
//     cublasDestroy(cublasH);
//     cusolverDnDestroy(cusolverH);
//     return 0;
// }

int main(int argc, char* argv[])
{   
    cudaDeviceSynchronize();

    // string file_name = "data/processed/mnist.csv";
    string file_name = "data/processed/mnist.csv";
    arma::arma_rng::set_seed_random();

    arma::mat A;
    A.load(file_name, arma::csv_ascii);
    int d = A.n_cols;
    arma::mat X = A.cols(0, d-2);
    arma::vec Y = A.col(d-1);
    arma::uvec uY = arma::conv_to<arma::uvec>::from(Y);

    arma::uvec inds = arma::find(uY == 0);
    for (int i = 1; i < 5; ++i) 
    {
        inds = arma::join_cols(inds, arma::find(uY == i));
    }

    std::sort(inds.begin(), inds.end());
    X = X.rows(inds);
    uY = uY.rows(inds);
    

    cout << X.n_rows << ", " << X.n_cols << endl;
    cout << uY.n_rows << ", " << uY.n_cols << endl;
    uY = uY - 1;
    cout << arma::unique(uY).t() << endl;

    unsigned int m = 1000;
    unsigned int k = 5;
    double accur;
    arma::uvec y_hat;

    cout << "In memory" << endl;
    CSSC cssc_clustering(X, k, m);
    tester(cssc_clustering, uY, "cssc");
    
    cout << "From file" << endl;
    MemCSSC mem_cssc_clustering(file_name, k, m);
    tester(mem_cssc_clustering, uY, "mem_cssc");
    

    cssc_clustering.fit();
    y_hat = cssc_clustering.get_y_hat();

    accur = cssc_clustering.accuracy(uY, y_hat);
    cout << endl;
    cout << "Accuracy: " << accur << endl;
    cout << endl;

    // Read and run MemCSSC on  MNIST8M dataset
    // file_name = "data/processed/mnist8m_subset.csv";
    // // Read data/processed/mnist8m.csv line by line, taking last element as label and 
    // // element of arma::uvec
    // cout << "--------------------------" << endl;
    // cout << "MNIST8M" << endl;
    // cout << "--------------------------" << endl;
    // read_labels(file_name, uY);
    // // uY.save("mnist8m_Y.csv", arma::csv_ascii);

    // MemCSSC mem_cssc_clustering2(file_name, k, m);
    // tester(mem_cssc_clustering2, uY, "mem_cssc_mnist8m");
    return 0;
}