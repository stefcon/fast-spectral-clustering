#include "../lib/clustering.hpp"
#include <vector>
#include <cmath>


// Path: bs190253d/diplomski/src/clustering.cpp
double Clustering::accuracy(const arma::uvec& Y, const arma::uvec &y_hat)
{
        arma::uvec uniques = arma::unique(Y);
        unsigned int k = uniques.n_rows;
        unsigned int n = Y.n_rows;
        std::vector<unsigned int> perm(k, 0);
        for (unsigned int i=0; i<k; i++) perm[i] = i + 1;
        double res = 0.0;
        do {
            arma::uvec yy(y_hat);
            for (unsigned int i=0; i<k; i++) {
            arma::uvec inds = arma::find(y_hat == i);
            arma::uvec f(inds.n_rows);
            f.fill(perm[i]);
            yy.rows(inds) = f;
            }
            arma::uvec match = yy == Y;
            double r = (double)arma::sum(match) / (double)n;
            res = fmax(r, res);
        } while (next_permutation(perm.begin(), perm.end()));

        return res;
}
