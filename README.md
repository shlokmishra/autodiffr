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

## Remaining Work

For the remainder of the semester, the following features are planned: implement `optim_mest()` for M-estimation with sandwich variance, add `vcov_info()` for information matrix diagnostics, create `profile_lik()` for profile likelihood confidence intervals with `autoplot()` visualization, develop `constraints()` helpers for smooth constraint handling (positive, simplex, correlation matrices), add `check_grad()` for gradient verification, implement `broom` methods (`tidy()`, `glance()`, `augment()`) for tidy output, create package vignettes with examples, and expand test coverage.

