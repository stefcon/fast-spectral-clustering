#include "../lib/sample.hpp"
#include <random>
#include <algorithm>
#include <vector>


arma::uvec sample_without_replacement(int low, int high, int n)
{
    std::vector<int> tmp;
    std::mt19937 gen = std::mt19937{std::random_device{}()};

    std::vector<int> v(high - low + 1);
    std::iota(v.begin(), v.end(), low);

    // From documentation:
    // Selects n elements from the sequence [first, last) (without replacement) such that each possible sample has
    // equal probability of appearance, and writes those selected elements into the output iterator out. Random numbers 
    // are generated using the random number generator g.
    std::sample(v.begin(), v.end(), std::back_inserter(tmp), n, gen);
    // Random reorder such that each possible permutati0on of those leements has qeual probability of appearance.
    std::shuffle(tmp.begin(), tmp.end(), gen);

    arma::uvec res = arma::conv_to<arma::uvec>::from(tmp);

    return res;
}