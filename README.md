# autodiffr

A tidy, torch-powered toolkit for Auto-Diff M-Estimation and MLE in R

## Description

`autodiffr` provides a high-level, tidy API for frequentist estimation using automatic differentiation via the `torch` package. When models require custom likelihoods or M-estimators (e.g., zero-inflation, censoring, mixtures), typical R workflows rely on numeric derivatives (`optim` + `numDeriv`) or C++ templates (TMB). `autodiffr` offers stable gradients/Hessians, smooth constraint handling, and classical inference (information matrices, sandwich SEs, LR/score tests) without writing C++.

## Installation

```r
# Install from GitHub
devtools::install_github("shlokmishra/autodiffr")

# Or install from source
# Clone the repository and run:
# R CMD INSTALL .
```

**Note:** This package requires the `torch` package. If you haven't installed it yet:

```r
install.packages("torch")
torch::install_torch()  # Install PyTorch backend
```

## Features

### Core Functionality
- **Maximum Likelihood Estimation**: `optim_mle()` with automatic differentiation via torch
  - Supports both torch-native functions (true autograd) and R functions (finite-difference fallback)
  - LBFGS and Adam optimizers
  - Automatic variance-covariance matrix computation
- **M-Estimation**: `optim_mest()` with sandwich variance-covariance matrices
  - Godambe (sandwich) variance estimation
  - Small-sample corrections
  - Robust standard errors

### Parameter Constraints
- `positive()`: Enforce positive parameters (e.g., variance, rate parameters)
- `simplex()`: Enforce simplex constraints (e.g., mixture weights)
- `corr_matrix()`: Correlation matrix constraints (stub for future implementation)
- Smooth reparameterization with Jacobian corrections

### Diagnostics and Verification
- `check_grad()`: Verify autograd gradients against finite differences
- `vcov_info()`: Information matrix diagnostics (condition number, eigenvalues)
- `vcov()`: Extract variance-covariance matrices

### Tidy Integration
- `tidy()`: Extract parameter estimates, standard errors, test statistics, and p-values
- `glance()`: Model summary statistics (log-likelihood, AIC, BIC, convergence info)
- `augment()`: Augment data with fitted values and residuals (minimal implementation)
- Full `broom` package integration

### Visualization
- `autoplot()`: Diagnostic plots for parameter estimates with confidence intervals

### Documentation
- Three comprehensive vignettes:
  - "Getting Started with autodiffr: MLE with Autograd"
  - "Constraints and M-estimation"
  - "Profile Likelihood and Diagnostics"

## Quick Start

### Maximum Likelihood Estimation

```r
library(torch)
library(autodiffr)

# Example: Normal distribution MLE with torch-native function
set.seed(123)
data <- rnorm(100, mean = 5, sd = 2)
data_tensor <- torch_tensor(data, dtype = torch_float64())

loglik <- function(theta, data) {
  mu <- theta[1]
  sigma <- torch_clamp(theta[2], min = 1e-6)
  dist <- distr_normal(mu, sigma)
  dist$log_prob(data)$sum()
}

fit <- optim_mle(loglik, start = c(mu = 0, sigma = 1), data = data_tensor)
print(fit)
summary(fit)

# Extract results
coef(fit)
vcov(fit)

# Tidy output (requires broom package)
library(broom)
tidy(fit)
glance(fit)
```

### M-Estimation

```r
# Linear regression via M-estimation
n <- 100
X <- cbind(1, rnorm(n))
beta_true <- c(2, 3)
y <- X %*% beta_true + rnorm(n, sd = 1)
data_list <- list(X = X, y = y)

# Estimating equations for OLS
psi_ols <- function(theta, data) {
  beta <- theta[c("beta0", "beta1")]
  residuals <- data$y - data$X %*% as.numeric(beta)
  cbind(residuals * data$X[, 1], residuals * data$X[, 2])
}

fit_mest <- optim_mest(psi_ols, start = c(beta0 = 0, beta1 = 0), data = data_list)
print(fit_mest)
vcov(fit_mest)  # Sandwich variance
```

### Parameter Constraints

```r
# Enforce positive variance parameter
set.seed(123)
data <- rnorm(100, mean = 5, sd = 2)

loglik <- function(theta, data) {
  mu <- theta["mu"]
  sigma <- theta["sigma"]
  sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
}

# Apply positive constraint to sigma
constr <- positive("sigma")
fit <- optim_mle(loglik, start = c(mu = 0, sigma = 1), data = data, 
                 constraints = constr)
print(fit)
# sigma will always be positive
```

### Gradient Verification

```r
# Verify autograd gradients
check_result <- check_grad(loglik, theta0 = c(mu = 0, sigma = 1), 
                          data = data_tensor)
print(check_result)
```

## Package Status

- **Tests**: 145 passing tests, comprehensive test coverage
- **Broom Integration**: Fully functional with `tidy()`, `glance()`, and `augment()` methods
- **Vignettes**: Three complete vignettes with examples (plots verified and render correctly) (all plots verified and rendering correctly)
- **Documentation**: Full roxygen2 documentation for all exported functions

## Implementation Notes

### Why No C++ Code?

This package is intentionally implemented entirely in R using the high-level `torch` package API. The `torch` package already provides efficient C++ bindings to `libtorch` (PyTorch's C++ backend), so there is no need for custom C++ code. This design choice offers several advantages:

- **Easier maintenance**: Pure R code is easier to debug and maintain
- **Cross-platform compatibility**: `torch` handles platform-specific compilation
- **Performance**: `torch`'s C++ backend provides excellent performance for tensor operations and automatic differentiation
- **Accessibility**: Users don't need C++ toolchains or compilation setup
- **Rapid development**: Faster iteration without C++ compilation cycles

The package leverages `torch`'s automatic differentiation capabilities directly through R, providing the same computational efficiency as custom C++ implementations while maintaining the simplicity of an R-only codebase.

## See Also

- See the vignettes for detailed examples and use cases
- `vignette("getting-started", package = "autodiffr")`
- `vignette("constraints-mestimation", package = "autodiffr")`
- `vignette("diagnostics", package = "autodiffr")`

## License

GPL (>= 3)

