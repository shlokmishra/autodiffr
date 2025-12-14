#' Check gradients computed by autograd
#'
#' Compares autograd gradients with finite differences to make sure your
#' log-likelihood function is implemented correctly. Useful for debugging
#' when things aren't working as expected.
#'
#' @param loglik A torch-native function that computes the log-likelihood. Must
#'   accept torch tensors as the first argument and return a scalar torch tensor.
#' @param theta0 Named numeric vector of parameter values at which to check gradients.
#' @param data The data object to be passed to `loglik`.
#' @param ... Additional arguments passed to `loglik`.
#' @param eps Step size for finite differences (default: 1e-6).
#' @param rel_tol Relative tolerance for considering gradients acceptable
#'   (default: 1e-3).
#'
#' @return An object of class `autodiffr_gradcheck` containing:
#'   \item{theta}{The parameter values at which gradients were checked}
#'   \item{grad_autograd}{Gradients computed via autograd}
#'   \item{grad_fd}{Gradients computed via finite differences}
#'   \item{abs_err}{Absolute errors between autograd and finite-difference gradients}
#'   \item{rel_err}{Relative errors between autograd and finite-difference gradients}
#'   \item{max_abs_err}{Maximum absolute error}
#'   \item{max_rel_err}{Maximum relative error}
#'   \item{ok}{Logical indicating if all relative errors are within tolerance}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(torch)
#' set.seed(123)
#' data <- rnorm(100, mean = 5, sd = 2)
#' data_tensor <- torch_tensor(data, dtype = torch_float64())
#'
#' # Torch-native loglikelihood
#' loglik_torch <- function(theta, data) {
#'   mu <- theta[1]
#'   sigma <- torch::torch_clamp(theta[2], min = 1e-6)
#'   dist <- torch::distr_normal(mu, sigma)
#'   dist$log_prob(data)$sum()
#' }
#'
#' theta0 <- c(mu = 5, sigma = 2)
#' check <- check_grad(loglik_torch, theta0, data_tensor)
#' print(check)
#' }
check_grad <- function(loglik,
                        theta0,
                        data,
                        ...,
                        eps = 1e-6,
                        rel_tol = 1e-3) {
  # Validate inputs
  if (!is.function(loglik)) {
    stop("loglik must be a function")
  }
  if (!is.numeric(theta0) || is.null(names(theta0)) || any(names(theta0) == "")) {
    stop("theta0 must be a named numeric vector")
  }
  if (length(theta0) == 0) {
    stop("theta0 must have at least one parameter")
  }

  # Check torch availability
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch package is required but not installed. ",
         "Install it with: install.packages('torch')")
  }

  param_names <- names(theta0)

  # Convert theta0 to torch tensor with gradient tracking
  theta_t <- torch::torch_tensor(
    as.numeric(theta0),
    requires_grad = TRUE,
    dtype = torch::torch_float64()
  )

  # Compute log-likelihood and get autograd gradient
  # Capture ... for use in loglik call
  dots <- list(...)
  tryCatch({
    # Call loglik with captured dots
    ll_tensor <- do.call(loglik, c(list(theta_t, data), dots))
    
    # Check that result is a scalar tensor
    if (ll_tensor$numel() != 1) {
      stop("loglik must return a scalar tensor")
    }
    
    # Compute gradients via autograd
    ll_tensor$backward()
    grad_autograd <- as.numeric(theta_t$grad$detach()$cpu())
    names(grad_autograd) <- param_names
  }, error = function(e) {
    stop("Error computing autograd gradient: ", conditionMessage(e),
         "\nMake sure loglik is a torch-native function that accepts torch tensors.")
  })

  # Compute finite-difference gradient
  # Use the existing finite_diff_grad function, but we need to wrap loglik
  # to work with numeric vectors
  # dots already captured above
  loglik_r_wrapper <- function(theta_r, data) {
    # Convert to torch tensor temporarily
    theta_tmp <- torch::torch_tensor(theta_r, dtype = torch::torch_float64())
    # Call loglik with captured dots
    ll_tmp <- do.call(loglik, c(list(theta_tmp, data), dots))
    as.numeric(ll_tmp$item())
  }
  
  grad_fd <- finite_diff_grad(loglik_r_wrapper, as.numeric(theta0), data, eps = eps)
  names(grad_fd) <- param_names

  # Compute errors
  abs_err <- abs(grad_autograd - grad_fd)
  names(abs_err) <- param_names
  
  # Relative error: abs_err / (abs(grad_fd) + tiny)
  tiny <- 1e-10
  rel_err <- abs_err / (abs(grad_fd) + tiny)
  names(rel_err) <- param_names

  # Find maximum errors
  max_abs_err <- max(abs_err, na.rm = TRUE)
  max_rel_err <- max(rel_err, na.rm = TRUE)

  # Check if gradients are acceptable
  ok <- is.finite(max_rel_err) && max_rel_err < rel_tol

  # Create and return gradcheck object
  result <- list(
    theta = theta0,
    grad_autograd = grad_autograd,
    grad_fd = grad_fd,
    abs_err = abs_err,
    rel_err = rel_err,
    max_abs_err = max_abs_err,
    max_rel_err = max_rel_err,
    ok = ok,
    rel_tol = rel_tol
  )
  
  class(result) <- "autodiffr_gradcheck"
  return(result)
}

