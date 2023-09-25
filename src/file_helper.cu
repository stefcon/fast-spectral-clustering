#include "../lib/files_helper.hpp"
#include "../lib/cuda_helper.h"
#include <fstream>
#include <iostream>


void get_data_dimensions(std::string& filepath, int& m, int& n) 
{
    m = 0;
    n = 0;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line))
    {
        m++;
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
    std::cout << "m: " << m << std::endl;
    std::cout << "n: " << n << std::endl;
}

void read_matrix_X_sparse(std::string& filepath, arma::mat& X)
{
    // Initialize X to zeros
    X.zeros();

    std::ifstream infile(filepath);
    std::string line;
    int i = 0;
    int curr_row = 0;
    while (std::getline(infile, line))
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
            X(curr_row, j) = val;
        }
        curr_row++;
    
        i++;
    }
    infile.close();
}