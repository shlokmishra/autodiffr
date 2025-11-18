#' Maximum Likelihood Estimation using Automatic Differentiation
#'
#' Maximize a user-supplied log-likelihood function using optimization via the
#' torch package. This function uses torch optimizers (LBFGS, Adam) for efficient
#' optimization. Note: Currently, gradients are computed using finite differences
#' since torch's autograd requires torch-native operations. Future versions will
#' support torch-native functions for true automatic differentiation.
#'
#' @param loglik A function that computes the log-likelihood. Must accept at least
#'   two arguments: `theta` (a named numeric vector of parameters) and `data`
#'   (the data object). Must return a scalar numeric value (the log-likelihood).
#' @param start A named numeric vector of starting values for the parameters.
#' @param data The data object to be passed to `loglik`.
#' @param optimizer Character string specifying the optimizer to use. Options:
#'   `"lbfgs"` (default) or `"adam"`.
#' @param max_iter Maximum number of iterations (default: 1000).
#' @param tolerance Convergence tolerance (default: 1e-6).
#' @param ... Additional arguments passed to the optimizer.
#'
#' @return An object of class `autodiffr_fit` containing:
#'   \item{coefficients}{Named numeric vector of parameter estimates}
#'   \item{loglik}{Final log-likelihood value}
#'   \item{convergence}{Convergence code (0 = success)}
#'   \item{message}{Convergence message}
#'   \item{iterations}{Number of iterations}
#'   \item{gradient_norm}{Norm of the final gradient}
#'   \item{gradient}{Named numeric vector of final gradients}
#'   \item{optimizer}{Character string naming the optimizer used}
#'   \item{call}{The original function call}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Example: MLE for normal distribution (mean and variance)
#' set.seed(123)
#' data <- rnorm(100, mean = 5, sd = 2)
#'
#' loglik <- function(theta, data) {
#'   mu <- theta["mu"]
#'   sigma <- theta["sigma"]
#'   sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
#' }
#'
#' start <- c(mu = 0, sigma = 1)
#' fit <- optim_mle(loglik, start, data)
#' print(fit)
#' }
optim_mle <- function(loglik,
                       start,
                       data,
                       optimizer = "lbfgs",
                       max_iter = 1000,
                       tolerance = 1e-6,
                       ...) {
  # Store the call
  call <- match.call()

  # Validate inputs
  validate_loglik(loglik)
  validate_start(start)

  # Check torch availability
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch package is required but not installed. ",
         "Install it with: install.packages('torch')")
  }

  # Convert start to torch tensors
  param_names <- names(start)
  n_params <- length(start)

  # Create torch tensors for parameters
  # We'll use a list to track individual parameter tensors
  params_list <- lapply(seq_along(start), function(i) {
    torch::torch_tensor(
      start[i],
      requires_grad = TRUE,
      dtype = torch::torch_float64()
    )
  })
  names(params_list) <- param_names

  # Create a closure that wraps the user's loglik function
  # Note: Since user function is R-based, we convert to R, call it, then convert back
  # This breaks autograd, but we'll compute gradients manually using torch's autograd
  # on a torch-native approximation or use finite differences
  # For now, we'll use a manual gradient computation approach
  compute_loss_and_grad <- function() {
    # Convert torch tensors to R numeric vector
    theta_r <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
    names(theta_r) <- param_names

    # Call user's loglik function
    ll_value <- loglik(theta_r, data)

    # Convert to torch tensor (negative because we'll minimize)
    loss_tensor <- torch::torch_tensor(-ll_value, dtype = torch::torch_float64(),
                                       requires_grad = TRUE)

    # Compute gradients using finite differences (since R function breaks autograd)
    # This is a temporary solution - future versions will support torch-native functions
    eps <- 1e-5
    grads <- numeric(length(params_list))
    for (i in seq_along(params_list)) {
      theta_plus <- theta_r
      theta_plus[i] <- theta_plus[i] + eps
      ll_plus <- loglik(theta_plus, data)
      grads[i] <- (ll_plus - ll_value) / eps
    }

    # Store gradients in parameter tensors
    for (i in seq_along(params_list)) {
      if (!is.null(params_list[[i]]$grad)) {
        params_list[[i]]$grad$zero_()
      }
      params_list[[i]]$grad <- torch::torch_tensor(-grads[i], dtype = torch::torch_float64())
    }

    return(loss_tensor)
  }

  # Set up optimizer - use Adam as it's more robust for manual gradients
  # LBFGS requires exact gradients which we're approximating
  if (optimizer == "lbfgs") {
    # For LBFGS, we need to provide exact gradients
    # Since we're using finite differences, we'll use a custom approach
    opt <- torch::optim_lbfgs(
      params_list,
      lr = 1,
      max_iter = max_iter,
      tolerance_grad = tolerance,
      tolerance_change = tolerance,
      ...
    )
  } else if (optimizer == "adam") {
    opt <- torch::optim_adam(params_list, lr = 0.01, ...)
  } else {
    stop("Only 'lbfgs' and 'adam' optimizers are currently supported")
  }

  # Track iteration count
  iter_count <- 0L
  converged <- FALSE
  conv_message <- ""
  last_loss <- Inf

  # Optimization loop
  closure_fn <- function() {
    iter_count <<- iter_count + 1L
    opt$zero_grad()
    loss <- compute_loss_and_grad()
    loss_value <- as.numeric(loss$item())
    
    # Check convergence
    if (abs(last_loss - loss_value) < tolerance) {
      converged <<- TRUE
      conv_message <<- "Converged (tolerance reached)"
    }
    last_loss <<- loss_value
    
    return(loss)
  }

  # Run optimization
  tryCatch({
    if (optimizer == "lbfgs") {
      opt$step(closure_fn)
    } else {
      # For Adam, run multiple steps
      for (i in seq_len(max_iter)) {
        closure_fn()
        opt$step()
        if (converged) break
      }
    }
    if (!converged && iter_count < max_iter) {
      converged <- TRUE
      conv_message <- "Optimization completed"
    } else if (!converged) {
      conv_message <- paste("Reached maximum iterations (", max_iter, ")", sep = "")
    }
  }, error = function(e) {
    conv_message <<- paste("Optimization error:", conditionMessage(e))
  })

  # Extract final values
  final_params <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
  names(final_params) <- param_names

  # Compute final log-likelihood
  theta_final <- final_params
  final_ll <- loglik(theta_final, data)

  # Extract gradients (recompute for final parameters)
  eps <- 1e-5
  final_grad <- numeric(length(params_list))
  for (i in seq_along(params_list)) {
    theta_plus <- theta_final
    theta_plus[i] <- theta_plus[i] + eps
    ll_plus <- loglik(theta_plus, data)
    final_grad[i] <- (ll_plus - final_ll) / eps
  }
  names(final_grad) <- param_names
  grad_norm <- sqrt(sum(final_grad^2))

  # Determine convergence code
  convergence_code <- if (converged && grad_norm < tolerance) {
    0L
  } else if (converged) {
    1L  # Converged but gradient not small enough
  } else {
    2L  # Did not converge
  }

  # Create and return fit object
  autodiffr_fit(
    coefficients = final_params,
    loglik = final_ll,
    convergence = convergence_code,
    message = conv_message,
    iterations = iter_count,
    gradient_norm = grad_norm,
    gradient = final_grad,
    optimizer = optimizer,
    call = call
  )
}

