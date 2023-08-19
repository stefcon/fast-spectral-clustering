#include "../lib/cssc.hpp"
#include <iostream>
#include <string>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include "../lib/sample.hpp"

using namespace std;

int main(int argc, char* argv[])
{   

    cudaDeviceSynchronize();
    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;
    cublasCreate(&cublasH);
    cusolverDnCreate(&cusolverH);

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

    X = X.rows(inds);
    uY = uY.rows(inds);

    cout << X.n_rows << ", " << X.n_cols << endl;
    cout << uY.n_rows << ", " << uY.n_cols << endl;
    cout << arma::unique(uY).t() << endl;

    unsigned int m = 500;
    unsigned int k = 5;
    CSSC cssc_clustering(X, k, m);
    cssc_clustering.gpu_fit(cublasH, cusolverH);
    arma::uvec y_hat_gpu = cssc_clustering.get_y_hat();

    double accur_gpu = cssc_clustering.accuracy(uY, y_hat_gpu);
    cout << endl;
    cout << "Accuracy: " << accur_gpu << endl;
    cout << endl;

    // cssc_clustering.fit();
    // arma::uvec y_hat = cssc_clustering.get_y_hat();

    // double accur = cssc_clustering.accuracy(uY, y_hat);
    // cout << endl;
    // cout << "Accuracy: " << accur<< endl;
    // cout << endl;

    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    return 0;
}