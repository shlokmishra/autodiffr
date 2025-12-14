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

- **Maximum Likelihood Estimation**: `optim_mle()` with automatic differentiation via torch
- **M-Estimation**: `optim_mest()` with sandwich variance-covariance matrices
- **Parameter Constraints**: `positive()`, `simplex()`, `corr_matrix()` for smooth constraint handling
- **Gradient Verification**: `check_grad()` to verify autograd gradients
- **Variance Diagnostics**: `vcov_info()` for information matrix diagnostics
- **Broom Integration**: `tidy()`, `glance()`, and `augment()` methods for tidy output
- **Visualization**: `autoplot()` for diagnostic plots
- **Comprehensive Vignettes**: Three detailed vignettes with examples

## Quick Start

```r
library(torch)
library(autodiffr)

# Example: Normal distribution MLE
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
```

See the vignettes for more detailed examples.

