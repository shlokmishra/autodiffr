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

torch_to_r <- function(x) {
  as.numeric(x$detach()$cpu())
}

extract_gradient <- function(x) {
  if (is.null(x$grad)) {
    return(NULL)
  }
  torch_to_r(x$grad)
}

torch_norm <- function(x) {
  as.numeric(x$norm()$item())
}

validate_loglik <- function(loglik) {
  if (!is.function(loglik)) {
    stop("loglik must be a function")
  }
  if (length(formals(loglik)) < 2) {
    stop("loglik must accept at least two arguments: theta and data")
  }
}

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

is_torch_native <- function(loglik, test_params, test_data) {
  tryCatch({
    result <- loglik(test_params, test_data)
    # Check if result is a torch tensor
    inherits(result, "torch_tensor")
  }, error = function(e) {
    FALSE
  })
}

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


