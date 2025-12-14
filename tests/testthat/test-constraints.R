test_that("positive constraint works", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(100, mean = 5, sd = 2)
  
  # Log-likelihood for normal with positive variance constraint
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]  # Should be positive
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  constr <- positive("sigma")
  
  fit <- optim_mle(loglik_r, start, data, constraints = constr, 
                   optimizer = "adam", max_iter = 100)
  
  # Check structure
  expect_s3_class(fit, "autodiffr_fit")
  expect_named(fit$coefficients, c("mu", "sigma"))
  
  # Check that sigma is always positive (main constraint requirement)
  expect_true(fit$coefficients["sigma"] > 0)
  
  # Check that estimates are finite and reasonable
  expect_true(is.finite(fit$coefficients["mu"]))
  expect_true(is.finite(fit$coefficients["sigma"]))
  expect_true(fit$coefficients["sigma"] > 0.01)  # Must be positive and reasonable
})

test_that("simplex constraint works", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  # Create mixture data
  n <- 100
  weights_true <- c(0.3, 0.5, 0.2)
  data <- c(
    rnorm(n * weights_true[1], mean = 0, sd = 1),
    rnorm(n * weights_true[2], mean = 3, sd = 1),
    rnorm(n * weights_true[3], mean = 6, sd = 1)
  )
  data <- sample(data)  # Shuffle
  
  # Log-likelihood for mixture model with simplex weights
  loglik_r <- function(theta, data) {
    w1 <- theta["w1"]
    w2 <- theta["w2"]
    w3 <- theta["w3"]
    mu1 <- theta["mu1"]
    mu2 <- theta["mu2"]
    mu3 <- theta["mu3"]
    
    # Mixture log-likelihood
    ll <- sum(log(
      w1 * dnorm(data, mu1, 1) +
      w2 * dnorm(data, mu2, 1) +
      w3 * dnorm(data, mu3, 1)
    ))
    if (!is.finite(ll)) return(-1e10)
    ll
  }
  
  start <- c(w1 = 0.33, w2 = 0.33, w3 = 0.34, mu1 = 0, mu2 = 3, mu3 = 6)
  constr <- simplex("w1", k = 3)
  
  fit <- optim_mle(loglik_r, start, data, constraints = constr,
                   optimizer = "adam", max_iter = 150)
  
  # Check structure
  expect_s3_class(fit, "autodiffr_fit")
  
  # Check that weights are non-negative and sum to 1
  w1 <- fit$coefficients["w1"]
  w2 <- fit$coefficients["w2"]
  w3 <- fit$coefficients["w3"]
  
  expect_true(w1 >= 0 && w2 >= 0 && w3 >= 0)
  expect_true(abs(w1 + w2 + w3 - 1) < 1e-6)
})

test_that("constraints() helper works", {
  skip_if_not_installed("torch")
  
  constr1 <- positive("sigma")
  constr2 <- simplex("weights", k = 3)
  
  constr_list <- constraints(constr1, constr2)
  
  expect_s3_class(constr_list, "autodiffr_constraints")
  expect_length(constr_list, 2)
  expect_s3_class(constr_list[[1]], "autodiffr_constraint")
  expect_s3_class(constr_list[[2]], "autodiffr_constraint")
})

test_that("corr_matrix constraint gives informative error", {
  expect_error(
    corr_matrix("R", dim = 3),
    "corr_matrix constraints are not yet implemented"
  )
})

test_that("constraints validation works", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50, mean = 3, sd = 1.5)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = 1)
  
  # Invalid constraint
  expect_error(
    optim_mle(loglik_r, start, data, constraints = "not a constraint"),
    "constraints must be a constraint object"
  )
  
  # Constraint with wrong parameter name
  constr_wrong <- positive("wrong_name")
  expect_error(
    optim_mle(loglik_r, start, data, constraints = constr_wrong),
    "Parameter 'wrong_name' not found"
  )
})

test_that("positive constraint with negative start value fails", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50, mean = 3, sd = 1.5)
  
  loglik_r <- function(theta, data) {
    mu <- theta["mu"]
    sigma <- theta["sigma"]
    sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
  }
  
  start <- c(mu = 0, sigma = -1)  # Negative start
  constr <- positive("sigma")
  
  expect_error(
    optim_mle(loglik_r, start, data, constraints = constr),
    "Starting value for 'sigma' must be positive"
  )
})

test_that("simplex constraint with invalid start fails", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data <- rnorm(50)
  
  loglik_r <- function(theta, data) {
    w1 <- theta["w1"]
    w2 <- theta["w2"]
    w3 <- theta["w3"]
    sum(log(w1 * dnorm(data, 0, 1) + w2 * dnorm(data, 1, 1) + w3 * dnorm(data, 2, 1)))
  }
  
  # Start that doesn't sum to 1
  start <- c(w1 = 0.5, w2 = 0.5, w3 = 0.5)  # Sums to 1.5
  constr <- simplex("w1", k = 3)
  
  expect_error(
    optim_mle(loglik_r, start, data, constraints = constr),
    "Starting values for simplex.*must.*sum to 1"
  )
})

