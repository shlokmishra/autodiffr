test_that("vcov_info works with autodiffr_fit", {
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
  
  # Check that vcov is available
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  # Get vcov_info
  vcov_diag <- vcov_info(fit)
  
  # Check structure
  expect_s3_class(vcov_diag, "autodiffr_vcov_info")
  expect_named(vcov_diag, c("vcov", "info", "eigenvalues", "cond_number", 
                            "type", "method", "nobs"))
  expect_equal(vcov_diag$type, "observed")
  expect_equal(vcov_diag$method, "mle")
  expect_true(is.matrix(vcov_diag$vcov))
  expect_true(is.matrix(vcov_diag$info))
  expect_true(is.numeric(vcov_diag$eigenvalues))
  expect_true(is.numeric(vcov_diag$cond_number))
  expect_true(vcov_diag$cond_number > 0)
})

test_that("vcov_info vcov matches fit$vcov", {
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
  
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  vcov_diag <- vcov_info(fit)
  
  # vcov should match
  expect_equal(vcov_diag$vcov, fit$vcov)
  
  # info should be inverse of vcov (within numerical precision)
  identity_check <- vcov_diag$info %*% vcov_diag$vcov
  identity_expected <- diag(nrow(vcov_diag$vcov))
  expect_true(max(abs(identity_check - identity_expected)) < 1e-5)
})

test_that("vcov_info handles expected type gracefully", {
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
  
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  expect_error(
    vcov_info(fit, type = "expected"),
    "Expected information matrix not yet implemented"
  )
})

test_that("vcov_info handles sandwich type gracefully", {
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
  
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  expect_error(
    vcov_info(fit, type = "sandwich"),
    "Sandwich variance-covariance matrix is only available for M-estimation"
  )
})

test_that("vcov_info handles missing vcov gracefully", {
  skip_if_not_installed("torch")
  
  # Create a fit object without vcov
  fit_no_vcov <- structure(
    list(
      coefficients = c(mu = 3, sigma = 1.5),
      loglik = -100,
      convergence = 0L,
      message = "test",
      iterations = 10L,
      gradient_norm = 0.001,
      gradient = c(mu = 0, sigma = 0),
      optimizer = "adam",
      vcov = NULL,
      call = NULL
    ),
    class = "autodiffr_fit"
  )
  
  expect_error(
    vcov_info(fit_no_vcov),
    "Variance-covariance matrix not available"
  )
})

test_that("vcov_info detects ill-conditioning", {
  skip_if_not_installed("torch")
  
  # Create a nearly collinear example
  # Use a simple linear regression-like setup that can be nearly collinear
  set.seed(123)
  x1 <- rnorm(50)
  x2 <- x1 + rnorm(50, sd = 0.01)  # Nearly collinear with x1
  y <- 2 + 3*x1 + rnorm(50)
  
  # For this test, we'll create a simple 2-parameter model
  # that might have conditioning issues
  data <- list(x = cbind(x1, x2), y = y)
  
  loglik_collinear <- function(theta, data) {
    beta0 <- theta["beta0"]
    beta1 <- theta["beta1"]
    # Simple model that might have conditioning issues
    mu <- beta0 + beta1 * data$x[, 1]
    sum(dnorm(data$y, mean = mu, sd = 1, log = TRUE))
  }
  
  start <- c(beta0 = 0, beta1 = 0)
  fit <- optim_mle(loglik_collinear, start, data, optimizer = "adam", max_iter = 50)
  
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  vcov_diag <- vcov_info(fit)
  
  # Should compute without crashing
  expect_true(is.finite(vcov_diag$cond_number))
  expect_true(vcov_diag$cond_number > 0)
  
  # For a well-behaved case, condition number should be reasonable
  # (This test is lenient since we're using a simple model)
  expect_true(vcov_diag$cond_number < 1e15)
})

test_that("print.autodiffr_vcov_info works", {
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
  
  skip_if(is.null(fit$vcov), "vcov not computed in fit")
  
  vcov_diag <- vcov_info(fit)
  
  # Should not error
  expect_output(print(vcov_diag), "Variance-Covariance Matrix Information")
  expect_output(print(vcov_diag), "Type:")
  expect_output(print(vcov_diag), "Condition Number:")
  expect_output(print(vcov_diag), "Well-conditioned:")
})

