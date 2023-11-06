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
    /*
    Run NUM_ITER iterations of clustering and store the labels in a vector.
    Calculate the mean and variance of the accuracies.
    Args:
        clustering: Clustering object
        uY: arma::uvec of labels
        label_folder_name: name of folder to store labels in
    */
    vector<double> accuracies;
    vector<arma::uvec> labels;
    double accur;
    arma::uvec y_hat;
    for (int iter = 0; iter < NUM_ITER; ++iter) {
        cout << "Iteration: " << iter << endl;
        clustering.fit();
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
    /*
    Run num_iter iterations of clustering and store the labels in a vector.
    Args:
        clustering: Clustering object
        label_folder_name: name of folder to store labels in
        num_iter: number of iterations to run
    */
    vector<arma::uvec> labels;
    arma::uvec y_hat;
    string folder_name = "data/processed/" + label_folder_name;
    fs::create_directories(folder_name);
    for (int iter = 0; iter < num_iter; ++iter) {
        clustering.fit();

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
    /*
    Read source_file line by line and write to target_file if the label is less than or equal to last_label argument
    Args:
        source_file: path to source file
        target_file: path to target file
        last_label: last label to include in target file
    */
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

vector<int> get_top_k_differences(arma::vec& vec, int k)
{
    /*
    Args:
        vec: arma::vec
        k: number of top differences to return
    Return:
        vector of indices of top k differences
    */
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

void example1()
{
    /*
    Running CSSC and MemCSSC on MNIST dataset
    */
    cudaDeviceSynchronize();

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
    cout << arma::unique(uY).t() << endl;

    unsigned int m = 1000;
    unsigned int k = 5;
    double accur;
    arma::uvec y_hat;

    cout << "Memory optimized" << endl;
    int x_n, n;
    get_data_dimensions_csv(file_name, x_n, n);
    cout << x_n << ", " << n << endl;
    MemCSSC mem_cssc_clustering(file_name, inds, n, k, m);
    tester(mem_cssc_clustering, uY, "mem_cssc");

    cout << "Sequential" << endl;
    CSSC cssc_clustering(X, k, m);
    tester(cssc_clustering, uY, "cssc");
}

void example2()
{
    /*
    Running MemCSSC on StereoSeq mouse brain dataset with different values of k.
    All results are stored in folders in data/processed.
    */
    cudaDeviceSynchronize();

    string file_name = "data/processed/Data_C5.npzcoo.csv";
    arma::arma_rng::set_seed_random();
    
    int m = 1000;
    int x_n, n;
    get_data_dimensions_coo(file_name, x_n, n);
    for (int k = 2; k <= 10; k++) {
        SparseMemCSSC mem_cssc_clustering(file_name, x_n, n, k, m);
        runner(mem_cssc_clustering, "mem_cssc_bgi/" + std::to_string(k), 4);
    }
}

void example3()
{
    cudaDeviceSynchronize();

    // string file_name = "data/processed/usps.csv";
    string file_name = "data/processed/Data_C5.npzcoo.csv";

    int m, n;
    arma::arma_rng::set_seed_random();

    // arma::vec eigvals;
    // eigvals.load("eigvals.csv", arma::csv_ascii);
    // auto inds = get_top_k_differences(eigvals, 15);
    // for (auto ind : inds) {
    //     cout << ind << ", ";
    // }

    int k = 50;

    get_data_dimensions_coo(file_name, m, n);
    arma::mat X(m,n);
    read_matrix_X_sparse_coo(file_name, X);
    // ---------------------------
    // arma::mat X;
    // X.load(file_name, arma::csv_ascii);
    
    arma::vec eigvals(k);
    int result = find_optimal_k(X, k, eigvals);
    cout << "Optimal k: " << result << endl;
}

void example4()
{
    cudaDeviceSynchronize();

    int x_n, n;
    arma::uvec uY;
    unsigned int m = 1000;
    unsigned int k = 5;
    double accur;
    arma::uvec y_hat;

    // Read and run MemCSSC on  MNIST8M dataset
    string file_name = "data/processed/mnist8m_subset.csv";
    // Read data/processed/mnist8m.csv line by line, taking last element as 
    // label and  element of arma::uvec
    cout << "--------------------------" << endl;
    cout << "MNIST8M" << endl;
    cout << "--------------------------" << endl;
    read_labels(file_name, uY);
    uY.save("mnist8m_Y.csv", arma::csv_ascii);


    MemCSSC mem_cssc_clustering2(file_name, MNIST8M_SUBSET_SIZE, MNIST8M_DIM, k, m);
    tester(mem_cssc_clustering2, uY, "mem_cssc_mnist8m");
}

int main(int argc, char* argv[])
{   

    // example1();
    // example2();
    // example3();
    example4();

    return 0;
}
