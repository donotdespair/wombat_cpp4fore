#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;


// [[Rcpp::export]]
vec rw (
    int T
) {
  return cumsum(randn(T));
} // END rw



/*** R
plot.ts(rw(100))
*/