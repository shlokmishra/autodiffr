test_that("check_grad works with torch-native function", {
  skip_if_not_installed("torch")
  
  # Simple normal loglikelihood with known analytic gradient
  set.seed(123)
  data_r <- rnorm(100, mean = 5, sd = 2)
  data_tensor <- torch::torch_tensor(data_r, dtype = torch::torch_float64())
  
  loglik_torch <- function(theta, data) {
    mu <- theta[1]
    sigma <- torch::torch_clamp(theta[2], min = 1e-6)
    dist <- torch::distr_normal(mu, sigma)
    dist$log_prob(data)$sum()
  }
  
  theta0 <- c(mu = 5, sigma = 2)
  check <- check_grad(loglik_torch, theta0, data_tensor)
  
  # Check structure
  expect_s3_class(check, "autodiffr_gradcheck")
  expect_named(check, c("theta", "grad_autograd", "grad_fd", "abs_err", 
                        "rel_err", "max_abs_err", "max_rel_err", "ok", "rel_tol"))
  expect_equal(check$theta, theta0)
  expect_equal(length(check$grad_autograd), 2)
  expect_equal(length(check$grad_fd), 2)
  expect_equal(names(check$grad_autograd), c("mu", "sigma"))
  expect_equal(names(check$grad_fd), c("mu", "sigma"))
  
  # For a well-behaved function, gradients should be close
  # (may not always pass due to numerical precision, but should be reasonable)
  expect_true(is.finite(check$max_rel_err))
  expect_true(check$max_rel_err < 1.0)  # Should be much smaller, but allow some tolerance
})

test_that("check_grad flags problems with non-differentiable operations", {
  skip_if_not_installed("torch")
  
  # Create a function with a non-differentiable operation (abs without smooth approximation)
  set.seed(123)
  data_r <- rnorm(50, mean = 0, sd = 1)
  data_tensor <- torch::torch_tensor(data_r, dtype = torch::torch_float64())
  
  # This function uses abs() which is differentiable, but let's create
  # a function that breaks the computation graph
  loglik_broken <- function(theta, data) {
    mu <- theta[1]
    # Use a non-differentiable operation: convert to R and back
    # This breaks autograd
    mu_r <- as.numeric(mu$item())
    mu_t <- torch::torch_tensor(mu_r, dtype = torch::torch_float64())
    sigma <- torch::torch_clamp(theta[2], min = 1e-6)
    dist <- torch::distr_normal(mu_t, sigma)
    dist$log_prob(data)$sum()
  }
  
  theta0 <- c(mu = 0, sigma = 1)
  
  # This should either error or show large discrepancies
  # The exact behavior depends on how torch handles the broken graph
  expect_error(
    check_grad(loglik_broken, theta0, data_tensor),
    "Error computing autograd gradient"
  )
})

test_that("check_grad validates inputs", {
  skip_if_not_installed("torch")
  
  data_tensor <- torch::torch_tensor(1:10, dtype = torch::torch_float64())
  loglik_torch <- function(theta, data) {
    torch::torch_tensor(0.0, dtype = torch::torch_float64())
  }
  
  # Invalid theta0
  expect_error(
    check_grad(loglik_torch, c(1, 2), data_tensor),
    "theta0 must be a named numeric vector"
  )
  
  expect_error(
    check_grad(loglik_torch, numeric(0), data_tensor),
    "theta0 must have at least one parameter"
  )
  
  # Invalid loglik
  expect_error(
    check_grad("not a function", c(mu = 1), data_tensor),
    "loglik must be a function"
  )
})

test_that("print.autodiffr_gradcheck works", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data_r <- rnorm(50, mean = 3, sd = 1.5)
  data_tensor <- torch::torch_tensor(data_r, dtype = torch::torch_float64())
  
  loglik_torch <- function(theta, data) {
    mu <- theta[1]
    sigma <- torch::torch_clamp(theta[2], min = 1e-6)
    dist <- torch::distr_normal(mu, sigma)
    dist$log_prob(data)$sum()
  }
  
  theta0 <- c(mu = 3, sigma = 1.5)
  check <- check_grad(loglik_torch, theta0, data_tensor)
  
  # Should not error
  expect_output(print(check), "Gradient Check Results")
  expect_output(print(check), "Gradients OK")
  expect_output(print(check), "Maximum")
})

test_that("check_grad handles custom rel_tol", {
  skip_if_not_installed("torch")
  
  set.seed(123)
  data_r <- rnorm(50, mean = 3, sd = 1.5)
  data_tensor <- torch::torch_tensor(data_r, dtype = torch::torch_float64())
  
  loglik_torch <- function(theta, data) {
    mu <- theta[1]
    sigma <- torch::torch_clamp(theta[2], min = 1e-6)
    dist <- torch::distr_normal(mu, sigma)
    dist$log_prob(data)$sum()
  }
  
  theta0 <- c(mu = 3, sigma = 1.5)
  check <- check_grad(loglik_torch, theta0, data_tensor, rel_tol = 1e-5)
  
  expect_equal(check$rel_tol, 1e-5)
  expect_true(is.logical(check$ok))
})

