#' Convert R numeric vector to torch tensor with gradient tracking
#'
#' @param x Named numeric vector
#' @param requires_grad Logical, whether to track gradients
#'
#' @return A torch tensor
#'
#' @keywords internal
r_to_torch <- function(x, requires_grad = TRUE) {
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch package is required but not installed")
  }
  torch::torch_tensor(
    as.numeric(x),
    requires_grad = requires_grad,
    dtype = torch::torch_float64()
  )
}

#' Convert torch tensor to R numeric vector
#'
#' @param x Torch tensor
#'
#' @return Numeric vector
#'
#' @keywords internal
torch_to_r <- function(x) {
  as.numeric(x$detach()$cpu())
}

#' Extract gradient from torch tensor
#'
#' @param x Torch tensor with gradient
#'
#' @return Numeric vector of gradients, or NULL if no gradient
#'
#' @keywords internal
extract_gradient <- function(x) {
  if (is.null(x$grad)) {
    return(NULL)
  }
  torch_to_r(x$grad)
}

#' Compute norm of a torch tensor
#'
#' @param x Torch tensor
#'
#' @return Numeric scalar
#'
#' @keywords internal
torch_norm <- function(x) {
  as.numeric(x$norm()$item())
}

#' Validate that loglik is a function
#'
#' @param loglik Object to validate
#'
#' @return Invisible NULL, throws error if invalid
#'
#' @keywords internal
validate_loglik <- function(loglik) {
  if (!is.function(loglik)) {
    stop("loglik must be a function")
  }
  if (length(formals(loglik)) < 2) {
    stop("loglik must accept at least two arguments: theta and data")
  }
}

#' Validate that start is a named numeric vector
#'
#' @param start Object to validate
#'
#' @return Invisible NULL, throws error if invalid
#'
#' @keywords internal
validate_start <- function(start) {
  if (!is.numeric(start)) {
    stop("start must be a numeric vector")
  }
  if (is.null(names(start)) || any(names(start) == "")) {
    stop("start must be a named numeric vector")
  }
  if (length(start) == 0) {
    stop("start must have at least one parameter")
  }
}

#' Check if a function is torch-native (accepts torch tensors)
#'
#' @param loglik The loglikelihood function to test
#' @param test_params Test parameters (torch tensors)
#' @param test_data Test data
#'
#' @return Logical, TRUE if function accepts torch tensors
#'
#' @keywords internal
is_torch_native <- function(loglik, test_params, test_data) {
  tryCatch({
    result <- loglik(test_params, test_data)
    # Check if result is a torch tensor
    inherits(result, "torch_tensor")
  }, error = function(e) {
    FALSE
  })
}

#' Compute gradients using finite differences
#'
#' @param loglik Log-likelihood function (R-based)
#' @param theta Parameter vector (R numeric)
#' @param data Data object
#' @param eps Step size for finite differences
#'
#' @return Numeric vector of gradients
#'
#' @keywords internal
finite_diff_grad <- function(loglik, theta, data, eps = 1e-5) {
  n_params <- length(theta)
  grads <- numeric(n_params)
  ll_value <- loglik(theta, data)
  
  for (i in seq_len(n_params)) {
    theta_plus <- theta
    theta_plus[i] <- theta_plus[i] + eps
    ll_plus <- loglik(theta_plus, data)
    grads[i] <- (ll_plus - ll_value) / eps
  }
  
  return(grads)
}

#' Compute Hessian using finite differences
#'
#' @param loglik Log-likelihood function (R-based)
#' @param theta Parameter vector (R numeric)
#' @param data Data object
#' @param eps Step size for finite differences
#'
#' @return Numeric matrix (Hessian)
#'
#' @keywords internal
finite_diff_hessian <- function(loglik, theta, data, eps = 1e-5) {
  n_params <- length(theta)
  hessian <- matrix(0, nrow = n_params, ncol = n_params)
  
  # Compute gradients at theta
  grad_theta <- finite_diff_grad(loglik, theta, data, eps)
  
  # Compute second derivatives
  for (i in seq_len(n_params)) {
    theta_plus <- theta
    theta_plus[i] <- theta_plus[i] + eps
    grad_plus <- finite_diff_grad(loglik, theta_plus, data, eps)
    hessian[i, ] <- (grad_plus - grad_theta) / eps
  }
  
  # Symmetrize (should be symmetric but numerical errors may occur)
  hessian <- (hessian + t(hessian)) / 2
  
  return(hessian)
}


