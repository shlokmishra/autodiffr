---
title: "Getting Started with autodiffr: MLE with Autograd"
author: "Shlok Mishra"
date: "2025-12-14"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Getting Started with autodiffr: MLE with Autograd}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---



## Introduction

The `autodiffr` package provides automatic differentiation-based maximum likelihood estimation (MLE) and M-estimation in R using the `torch` package. This vignette demonstrates the basic workflow for fitting models with `optim_mle()`, extracting results with `vcov()`, and using `broom` methods for tidy output.

## Example 1: Normal Distribution MLE

Let's start with a simple example: estimating the mean and standard deviation of a normal distribution.

### Data Generation


``` r
set.seed(123)
n <- 100
true_mu <- 5
true_sigma <- 2
data <- rnorm(n, mean = true_mu, sd = true_sigma)
```

### Torch-Native Log-Likelihood Function

For best performance and accuracy, we'll write a torch-native log-likelihood function that uses `torch` operations:


``` r
library(torch)
library(autodiffr)
#> Error in library(autodiffr): there is no package called 'autodiffr'

# Convert data to torch tensor
data_tensor <- torch_tensor(data, dtype = torch_float64())

# Torch-native log-likelihood function
loglik_normal <- function(theta, data) {
  mu <- theta[1]
  sigma <- torch_clamp(theta[2], min = 1e-6)  # Ensure sigma > 0
  dist <- distr_normal(mu, sigma)
  dist$log_prob(data)$sum()
}
```

### Fitting the Model


``` r
# Starting values
start <- c(mu = 0, sigma = 1)

# Fit the model
fit <- optim_mle(
  loglik = loglik_normal,
  start = start,
  data = data_tensor,
  optimizer = "lbfgs",
  max_iter = 1000
)
#> Error in optim_mle(loglik = loglik_normal, start = start, data = data_tensor, : could not find function "optim_mle"

# View results
print(fit)
#> Error: object 'fit' not found
```

### Extracting Results


``` r
# Coefficients
coef(fit)
#> Error: object 'fit' not found

# Variance-covariance matrix
vcov(fit)
#> Error: object 'fit' not found

# Standard errors
sqrt(diag(vcov(fit)))
#> Error: object 'fit' not found

# Summary
summary(fit)
#> Error: object 'fit' not found
```

### Using Broom for Tidy Output


``` r
library(broom)

# Tidy output with estimates, SEs, statistics, and p-values
tidy(fit)
#> Error: object 'fit' not found

# One-row summary with model-level statistics
glance(fit)
#> Error: object 'fit' not found
```

## Example 2: Logistic Regression

Now let's fit a logistic regression model using MLE.

### Data Generation


``` r
set.seed(456)
n <- 200
X <- cbind(1, rnorm(n), rnorm(n))  # Design matrix with intercept
beta_true <- c(-1, 0.5, 1.2)
logit_p <- X %*% beta_true
p <- plogis(logit_p)
y <- rbinom(n, size = 1, prob = p)

# Convert to torch tensors
X_tensor <- torch_tensor(X, dtype = torch_float64())
y_tensor <- torch_tensor(y, dtype = torch_float64())
data_logistic <- list(X = X_tensor, y = y_tensor)
```

### Log-Likelihood Function


``` r
loglik_logistic <- function(theta, data) {
  beta <- theta[1:3]
  logit_p <- torch_matmul(data$X, beta)
  p <- torch_sigmoid(logit_p)
  
  # Log-likelihood: sum(y * log(p) + (1-y) * log(1-p))
  log_lik <- data$y * torch_log(p + 1e-8) + 
             (1 - data$y) * torch_log(1 - p + 1e-8)
  log_lik$sum()
}
```

### Fitting the Model


``` r
start_logistic <- c(beta0 = 0, beta1 = 0, beta2 = 0)

fit_logistic <- optim_mle(
  loglik = loglik_logistic,
  start = start_logistic,
  data = data_logistic,
  optimizer = "lbfgs"
)
#> Error in optim_mle(loglik = loglik_logistic, start = start_logistic, data = data_logistic, : could not find function "optim_mle"

print(fit_logistic)
#> Error: object 'fit_logistic' not found
```

### Comparing with glm()


``` r
# Compare with standard glm
glm_fit <- glm(y ~ X[, 2] + X[, 3], family = binomial)

# Our estimates
coef(fit_logistic)
#> Error: object 'fit_logistic' not found

# GLM estimates
coef(glm_fit)
#> (Intercept)      X[, 2]      X[, 3] 
#>   -1.217282    0.254441    1.130298

# Standard errors comparison
sqrt(diag(vcov(fit_logistic)))
#> Error: object 'fit_logistic' not found
sqrt(diag(vcov(glm_fit)))
#> (Intercept)      X[, 2]      X[, 3] 
#>   0.1962031   0.1711242   0.2183405
```

### Tidy Output


``` r
tidy(fit_logistic)
#> Error: object 'fit_logistic' not found
glance(fit_logistic)
#> Error: object 'fit_logistic' not found
```

## Example 3: R Function Mode (Fallback)

If you prefer to use standard R functions, `autodiffr` will automatically use finite differences:


``` r
# R function version
loglik_normal_r <- function(theta, data) {
  mu <- theta["mu"]
  sigma <- theta["sigma"]
  sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
}

fit_r <- optim_mle(
  loglik = loglik_normal_r,
  start = c(mu = 0, sigma = 1),
  data = data,
  optimizer = "lbfgs"
)
#> Error in optim_mle(loglik = loglik_normal_r, start = c(mu = 0, sigma = 1), : could not find function "optim_mle"

print(fit_r)
#> Error: object 'fit_r' not found
tidy(fit_r)
#> Error: object 'fit_r' not found
```

**Note:** Torch-native functions are recommended for better accuracy and performance, as they use exact gradients via autograd rather than finite differences.

## Summary

This vignette demonstrated:

- Writing torch-native log-likelihood functions
- Fitting models with `optim_mle()`
- Extracting coefficients and variance-covariance matrices
- Using `broom` methods for tidy output
- Comparing torch-native and R function modes

For more advanced features, see the other vignettes on constraints, M-estimation, and diagnostics.

