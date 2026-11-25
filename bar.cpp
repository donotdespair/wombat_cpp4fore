#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppTN.h>
// [[Rcpp::depends(RcppTN)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
vec rw (
    int T
) {
  return cumsum(randn(T));
} // END rw


// [[Rcpp::export]]
mat bar (
  int S,                  // number of posterior draws
  vec data,               // data vector (T+p)x1
  int p = 1               // number of autoregressive lags
) {
  // Bayesian autoregression: only autoregressive parameters are estimated

  int T           = data.n_rows - p;                                // read T
  mat V_inv       = diagmat(cumsum(ones<vec>(p)));              // create prior covariance matrix
  
  mat y           = data.rows(p, T + p - 1);                        // create data matrices
  mat x(T, p);
  for (int i = 0; i < p; i++) {
    x.col(i)      = data.rows(p - 1 - i, T + p - 2 - i);       // create lagged data matrices
  }
  
    // Exercise:
  //*********************************/
  // 1. Compute posterior covariance and mean for autoregressive parameters
  // 2. Sample random draws from the posterior distribution of the autoregressive parameters 
  // 3. Save the draws in an pxS matrix alpha_post and return it
  // mat alpha_post(S, p);                                   // matrix to store posterior draws

  // Delete the lines below in a version for students
  mat A           = V_inv + x.t() * x;                              // precision matrix of the posterior
  mat W           = inv(trimatu(chol(A)));                          // fast triangular inverse R^{-1}

  mat V_post      = W * W.t();                                 // posterior covariance matrix
  vec m_post      = V_post * x.t() * y;                        // posterior mean vector
  mat alpha_post  = mvnrnd( m_post, V_post, S );           // draw S samples from the posterior distribution

  return alpha_post;
} // end bar


// [[Rcpp::export]]
mat forecast (
  mat alpha_post,         // posterior draws of autoregressive parameters pxS
  vec data,               // data vector (T+p)x1
  int horizon = 1         // forecast horizon
) {
  // Forecasting with a Bayesian autoregression
  
  int S           = alpha_post.n_cols;                              // read dimensions
  int p           = alpha_post.n_rows;
  int T           = data.n_rows - p;

  mat forecasts(S, horizon);                              // matrix to store forecasts
  
  for (int s = 0; s < S; s++) {
    
    vec y_t       = reverse(data.subvec(T, T + p - 1));         // last p obs, most recent first

    for (int h = 0; h < horizon; h++) {
    
      double y_next   = randn( distr_param( dot(alpha_post.col(s), y_t), 1.0 ) );
      forecasts(s, h) = y_next;
      y_t             = join_cols( vec{y_next}, y_t.head(p - 1) );
    
    }
  }
  
  return forecasts;
} // end forecast




// [[Rcpp::export]]
mat forecast_truncated (
  mat alpha_post,         // posterior draws of autoregressive parameters pxS
  vec data,               // data vector (T+p)x1
  int horizon = 1,        // forecast horizon
  double lower_bound = NA_REAL,   // NA means no lower bound
  double upper_bound = NA_REAL    // NA means no upper bound
) {
  // Forecasting with a Bayesian autoregression

  if (std::isnan(lower_bound)) lower_bound = -datum::inf;
  if (std::isnan(upper_bound)) upper_bound =  datum::inf;

  int S           = alpha_post.n_cols;                              // read dimensions
  int p           = alpha_post.n_rows;
  int T           = data.n_rows - p;

  mat forecasts(S, horizon);                              // matrix to store forecasts
  
  for (int s = 0; s < S; s++) {
    
    vec y_t       = reverse(data.subvec(T, T + p - 1));         // last p obs, most recent first

    for (int h = 0; h < horizon; h++) {
    
      double y_next   = RcppTN::rtn1( dot(alpha_post.col(s), y_t), 1.0, lower_bound, upper_bound );
      forecasts(s, h) = y_next;
      y_t             = join_cols( vec{y_next}, y_t.head(p - 1) );
    
    }
  }
  
  return forecasts;
} // end forecast






/*** R
S       = 1000
p       = 4
T       = 200 + p
rw_data = rw(T)
post    = bar(S, rw_data, p)
fore    = forecast(post, rw_data, 10)
foretrunc = forecast_truncated(post, rw_data, 10, lower_bound = tail(rw_data,1))

# apply(post,2,mean)
# apply(post,2,sd)
# hist(apply(post,1,sum), breaks = 50)
plot.ts(c(
  tail(rw_data, 10),
  apply(fore, 2, mean)
))
lines(c(
  tail(rw_data, 10),
  apply(foretrunc, 2, mean)
))
*/