#include "../lib/kernels.hpp"
#include "../lib/timer.h"
#include "../lib/cuda_helper.h"
#include <cstdio>

void eig_dsymx_cusolver(
    cusolverDnHandle_t cusolverH, 
    double* d_A, 
    double* d_W,
    int m,
    int k, 
    double* d_eigvals,
    double* d_eigvecs
    )
{
    // Calculates eigenvalues and eigenvectors of a symmetric matrix using cusolver
    // Arguments:
    //      cusolverH: cusolver handle
    //      d_A: matrix to calculate eigenvalues and eigenvectors of
    //      d_W: array to store eigenvalues in
    //      m: size of the matrix
    //      k: number of eigenvalues to calculate
    //      d_eigvals: array to store eigenvalues in
    //      d_eigvecs: array to store eigenvectors in
    if (k == -1) {
        // Raise exception
        printf("eig_dsymx_cusolver: k must be value > 0!\n");                          \
        throw std::runtime_error("eig_dsymx_cusolver");
    }

    void* d_work;
    int* devInfo;
    int h_meig = k; // number of eigenvalues found in the interval
    int workspaceInBytes;

    
    CUSOLVER_CHECK(cusolverDnDsyevdx_bufferSize(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors
        CUSOLVER_EIG_RANGE_I,     // compute eigenvalues in an interval
        CUBLAS_FILL_MODE_LOWER,
        m,
        d_A,
        m,
        0.0, // vl - not used
        0.0, // vu - not used
        m-k+1,
        m,
        &h_meig,
        d_W,
        &workspaceInBytes
    ));
    
    // Initialize the workspace
    CUDA_CHECK(cudaMalloc(&d_work, workspaceInBytes * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&devInfo, sizeof(int)));


    CUSOLVER_CHECK(cusolverDnDsyevdx(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors.
        CUSOLVER_EIG_RANGE_I,     // compute eigenvalues in an interval
        CUBLAS_FILL_MODE_LOWER,
        m,      // size of the matrix
        d_A,    // matrix
        m,      // leading dimension of A
        0.0,    // vl - not used
        0.0,    // vu - not used
        m-k+1,    // il - lower bound of interval (index)
        m,      // iu - upper bound of interval (index)
        &h_meig,// number of eigenvalues found in the interval
        d_W,    // eigenvalues
        (double*)d_work,    // workspace
        workspaceInBytes,
        devInfo // error info
    ));
    

    // Initialize the eigenvalues and eigenvectors
    if (d_eigvals != nullptr)
        CUDA_CHECK(cudaMemcpy(d_eigvals, d_W, sizeof(double) * k, cudaMemcpyDeviceToDevice));
    if (d_eigvecs != nullptr)
        CUDA_CHECK(cudaMemcpy(d_eigvecs, d_A, sizeof(double) * m * k, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(d_work));
    CUDA_CHECK(cudaFree(devInfo));
}

void test_eig_kernel(arma::mat mat, int  m)
{
    cusolverDnHandle_t cusolverH;
    cusolverStatus_t cusolver_status;
    cusolverDnCreate(&cusolverH);
    double* d_A;
    double* d_W;
    void* d_work;
    int* devInfo;
    int h_meig; // number of eigenvalues found in the interval
    int workspaceInBytes;

    CUDA_CHECK(cudaMalloc(&d_A, sizeof(double) * m * m)); // Won't be needed if we use the matrix directly in the argument

    cusolver_status = cusolverDnDsyevdx_bufferSize(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors
        CUSOLVER_EIG_RANGE_I,     // compute eigenvalues in an interval
        CUBLAS_FILL_MODE_LOWER,
        m,
        d_A,
        m,
        0.0, // vl - not used
        0.0, // vu - not used
        m-1,
        m,
        &h_meig,
        d_W,
        &workspaceInBytes
    );
    // cudaMalloc(&d_W, m * sizeof(double));
    cudaMalloc(&d_work, workspaceInBytes * sizeof(double));
    cudaMalloc(&devInfo, sizeof(int));
    printf("Workspace size: %d\n", workspaceInBytes);
    int workspaceInBytes2;
    cusolver_status = cusolverDnDsyevd_bufferSize(
            cusolverH,
            CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors
            CUBLAS_FILL_MODE_LOWER,
            m,
            d_A,
            m,
            d_W,
            &workspaceInBytes2
    );
    printf("Workspace size: %d\n", workspaceInBytes2);

    cudaMemcpy(d_A, mat.memptr(), sizeof(double) * m * m, cudaMemcpyHostToDevice);

    cusolver_status = cusolverDnDsyevdx(
        cusolverH,
        CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors.
        CUSOLVER_EIG_RANGE_I,     // compute eigenvalues in an interval
        CUBLAS_FILL_MODE_LOWER,
        m,      // size of the matrix
        d_A, 
        m,      // leading dimension of A
        0.0,    // vl - not used
        0.0,    // vu - not used
        m-1,    // il - lower bound of interval (index)
        m,      // iu - upper bound of interval (index)
        &h_meig,// number of eigenvalues found in the interval
        d_W,    // eigenvalues
        (double*)d_work,    // workspace
        workspaceInBytes,
        devInfo // error info
    );

    // cusolver_status = cusolverDnDsyevd(
    //         cusolverH,
    //         CUSOLVER_EIG_MODE_VECTOR, // compute eigenvectors.
    //         CUBLAS_FILL_MODE_LOWER,
    //         m,      // size of the matrix
    //         d_A,    // matrix
    //         m,      // leading dimension of A
    //         d_W,    // eigenvalues
    //         (double*)d_work,    // workspace
    //         workspaceInBytes,
    //         devInfo // error info
    //     );

    // Check dev info
    int devInfo_h = 0;
    cudaMemcpy(&devInfo_h, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    printf("devInfo = %d\n", devInfo_h);
    arma::vec eigenvalues(m);
    cudaMemcpy(eigenvalues.memptr(), d_W, sizeof(double) * 2, cudaMemcpyDeviceToHost);
    eigenvalues.save("eigenvalues.txt", arma::raw_ascii);
    cudaMemcpy(mat.memptr(), d_A, sizeof(double) * m * m, cudaMemcpyDeviceToHost);
    mat.save("eigenvectors.txt", arma::raw_ascii);
    printf("h_meig = %d\n", h_meig);

    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_work);
    cudaFree(devInfo);
    cusolverDnDestroy(cusolverH);
}