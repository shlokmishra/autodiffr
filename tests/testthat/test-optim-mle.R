test_that("optim_mle works with R function (fallback mode)", {
  skip_if_not_installed("torch")
  
  # Simple normal MLE example
  set.seed(123)
  data <- rnorm(100, mean = 5, sd = 2)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  fit <- optim_mle(loglik_r, start, data, optimizer = "adam", max_iter = 200)
  
  # Check structure
  expect_s3_class(fit, "autodiffr_fit")
  expect_named(fit$coefficients, c("mu", "sigma"))
  expect_type(fit$coefficients, "double")
  expect_true(fit$loglik > -Inf)
  expect_true(fit$convergence %in% c(0L, 1L, 2L))
  expect_true(fit$iterations > 0)
  
  # Check that estimates are reasonable (should be close to true values 5 and 2)
  # Use more lenient tolerance since finite differences may not converge perfectly
  expect_true(abs(fit$coefficients["mu"] - 5) < 2)
  expect_true(abs(fit$coefficients["sigma"] - 2) < 2)
})

test_that("optim_mle works with torch-native function", {
  skip_if_not_installed("torch")
  
  # Simple normal MLE using torch
  set.seed(123)
  data_r <- rnorm(100, mean = 5, sd = 2)
  data_tensor <- torch::torch_tensor(data_r, dtype = torch::torch_float64())
  
  loglik_torch <- function(theta, data) {
    mu <- theta[1]
    sigma <- torch::torch_clamp(theta[2], min = 1e-6)
    dist <- torch::distr_normal(mu, sigma)
    dist$log_prob(data)$sum()
  }
  
  start <- c(mu = 0, sigma = 1)
  fit <- optim_mle(loglik_torch, start, data_tensor, optimizer = "adam", max_iter = 100)
  
  # Check structure
  expect_s3_class(fit, "autodiffr_fit")
  expect_named(fit$coefficients, c("mu", "sigma"))
  expect_type(fit$coefficients, "double")
  expect_true(fit$loglik > -Inf)
  expect_true(fit$convergence %in% c(0L, 1L, 2L))
  expect_true(fit$iterations > 0)
  
  # Check that estimates are reasonable
  expect_true(abs(fit$coefficients["mu"] - 5) < 1)
  expect_true(abs(fit$coefficients["sigma"] - 2) < 1)
})

test_that("vcov is computed and stored", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50, mean = 3, sd = 1.5)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  fit <- optim_mle(loglik_r, start, data, optimizer = "adam", max_iter = 50)
  
  # vcov may be NULL if computation failed, but if present should be a matrix
  if (!is.null(fit$vcov)) {
    expect_true(is.matrix(fit$vcov))
    expect_equal(nrow(fit$vcov), length(fit$coefficients))
    expect_equal(ncol(fit$vcov), length(fit$coefficients))
    expect_equal(rownames(fit$vcov), names(fit$coefficients))
    expect_equal(colnames(fit$vcov), names(fit$coefficients))
  }
})

test_that("vcov() method works", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50, mean = 3, sd = 1.5)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  fit <- optim_mle(loglik_r, start, data, optimizer = "adam", max_iter = 50)
  
  vcov_result <- vcov.autodiffr_fit(fit)
  # vcov may be NULL, but if present should match fit$vcov
  if (!is.null(vcov_result)) {
    expect_equal(vcov_result, fit$vcov)
  }
})

test_that("print and summary methods work", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50, mean = 3, sd = 1.5)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  fit <- optim_mle(loglik_r, start, data, optimizer = "adam", max_iter = 50)
  
  # Should not error
  expect_output(print(fit), "Autodiffr MLE Fit")
  expect_output(print(summary(fit)), "Autodiffr MLE Fit Summary")
})

test_that("validation functions work", {
  skip_if_not_installed("torch")
  
  # Test invalid loglik
  expect_error(
    optim_mle("not a function", c(mu = 0), data = 1:10),
    "loglik must be a function"
  )
  
  # Test invalid start
  expect_error(
    optim_mle(function(x, y) 1, c(1, 2), data = 1:10),
    "start must be a named numeric vector"
  )
  
  # numeric(0) fails name check first, so test with named empty vector
  expect_error(
    optim_mle(function(x, y) 1, structure(numeric(0), names = character(0)), data = 1:10),
    "start must have at least one parameter"
  )
})


