#include "../lib/sample.hpp"
#include <random>
#include <algorithm>
#include <filesystem>
#include <fstream>


std::vector<int> sample_without_replacement(int low, int high, int n)
{
    std::vector<int> inds;
    static std::mt19937 gen = std::mt19937{std::random_device{}()};

    std::vector<int> v(high - low + 1);
    std::iota(v.begin(), v.end(), low);

    // From documentation:
    // Selects n elements from the sequence [first, last) (without replacement) such that each possible sample has
    // equal probability of appearance, and writes those selected elements into the output iterator out. Random numbers 
    // are generated using the random number generator g.
    std::sample(v.begin(), v.end(), std::back_inserter(inds), n, gen);

    std::sort(inds.begin(), inds.end());

    return inds;
}


void read_labels(std::string& file_name, arma::uvec& uY)
{
    std::vector<int> labels;
    std::ifstream file(file_name);
    std::string line;
    while (std::getline(file, line)) {
        int label = line[line.length() - 1] - '0';
        labels.push_back(label);
    }
    uY = arma::conv_to<arma::uvec>::from(labels);
}
