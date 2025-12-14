#' Maximum Likelihood Estimation using Automatic Differentiation
#'
#' Maximize a user-supplied log-likelihood function using automatic differentiation
#' via the torch package. This function supports two modes:
#'
#' \strong{Torch-native mode (recommended):} For true automatic differentiation,
#' provide a `loglik` function that accepts torch tensors as the first argument
#' and uses torch operations internally. This enables exact gradient computation
#' via autograd.
#'
#' \strong{R function mode (fallback):} If you provide a standard R function that
#' accepts numeric vectors, gradients will be computed using finite differences.
#' This is less accurate and slower, but allows using existing R code.
#'
#' @param loglik A function that computes the log-likelihood. In torch-native mode,
#'   it must accept torch tensors as the first argument and return a torch tensor.
#'   In R function mode, it accepts a named numeric vector and returns a scalar.
#'   Must accept at least two arguments: `theta` (parameters) and `data`.
#' @param start A named numeric vector of starting values for the parameters.
#' @param data The data object to be passed to `loglik`.
#' @param optimizer Character string specifying the optimizer to use. Options:
#'   `"lbfgs"` (default) or `"adam"`.
#' @param max_iter Maximum number of iterations (default: 1000).
#' @param tolerance Convergence tolerance (default: 1e-6).
#' @param use_fallback Logical. If `TRUE`, force use of finite-difference gradients
#'   even for torch-native functions. If `FALSE` (default), automatically detect
#'   torch-native functions and use autograd when possible.
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
#'   \item{vcov}{Variance-covariance matrix (inverse of negative Hessian)}
#'   \item{optimizer}{Character string naming the optimizer used}
#'   \item{call}{The original function call}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Example: MLE for normal distribution using torch-native function
#' library(torch)
#' set.seed(123)
#' data <- rnorm(100, mean = 5, sd = 2)
#' data_tensor <- torch_tensor(data, dtype = torch_float64())
#'
#' # Torch-native loglikelihood (uses autograd)
#' loglik_torch <- function(theta, data) {
#'   mu <- theta[1]
#'   sigma <- torch::torch_clamp(theta[2], min = 1e-6)  # Ensure positive
#'   dist <- torch::distr_normal(mu, sigma)
#'   dist$log_prob(data)$sum()
#' }
#'
#' start <- c(mu = 0, sigma = 1)
#' fit <- optim_mle(loglik_torch, start, data_tensor)
#' print(fit)
#'
#' # R function mode (fallback, uses finite differences)
#' loglik_r <- function(theta, data) {
#'   mu <- theta["mu"]
#'   sigma <- theta["sigma"]
#'   sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
#' }
#'
#' fit2 <- optim_mle(loglik_r, start, data)
#' print(fit2)
#' }
optim_mle <- function(loglik,
                       start,
                       data,
                       optimizer = "lbfgs",
                       max_iter = 1000,
                       tolerance = 1e-6,
                       use_fallback = FALSE,
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
  params_list <- lapply(seq_along(start), function(i) {
    torch::torch_tensor(
      start[i],
      requires_grad = TRUE,
      dtype = torch::torch_float64()
    )
  })
  names(params_list) <- param_names

  # Create a single tensor for torch-native functions
  params_tensor <- torch::torch_tensor(
    as.numeric(start),
    requires_grad = TRUE,
    dtype = torch::torch_float64()
  )

  # Detect if function is torch-native (unless forced to use fallback)
  is_torch <- FALSE
  if (!use_fallback) {
    # Test with a torch tensor matching the number of parameters
    test_tensor <- torch::torch_tensor(
      rep(0.0, n_params),
      requires_grad = TRUE,
      dtype = torch::torch_float64()
    )
    is_torch <- is_torch_native(loglik, test_tensor, data)
  }

  if (is_torch) {
    # Torch-native mode: use true autograd
    compute_loss <- function() {
      # Call user's torch-native function
      ll_tensor <- loglik(params_tensor, data)
      
      # Ensure it's a scalar tensor
      if (ll_tensor$numel() != 1) {
        stop("loglik must return a scalar tensor")
      }
      
      # Return negative (we minimize)
      loss <- -ll_tensor
      return(loss)
    }
  } else {
    # R function mode: use finite differences
    compute_loss <- function() {
      # Convert torch tensors to R numeric vector
      theta_r <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
      names(theta_r) <- param_names

      # Call user's R function
      ll_value <- loglik(theta_r, data)

      # Convert to torch tensor (negative because we'll minimize)
      loss_tensor <- torch::torch_tensor(-ll_value, dtype = torch::torch_float64(),
                                         requires_grad = TRUE)

      # Compute gradients using finite differences
      eps <- 1e-5
      grads <- finite_diff_grad(loglik, theta_r, data, eps)

      # Store gradients in parameter tensors
      for (i in seq_along(params_list)) {
        if (!is.null(params_list[[i]]$grad)) {
          params_list[[i]]$grad$zero_()
        }
        params_list[[i]]$grad <- torch::torch_tensor(-grads[i], dtype = torch::torch_float64())
      }

      return(loss_tensor)
    }
  }

  # Set up optimizer
  if (optimizer == "lbfgs") {
    if (is_torch) {
      opt <- torch::optim_lbfgs(
        params_tensor,
        lr = 1,
        max_iter = max_iter,
        tolerance_grad = tolerance,
        tolerance_change = tolerance,
        ...
      )
    } else {
      opt <- torch::optim_lbfgs(
        params_list,
        lr = 1,
        max_iter = max_iter,
        tolerance_grad = tolerance,
        tolerance_change = tolerance,
        ...
      )
    }
  } else if (optimizer == "adam") {
    if (is_torch) {
      opt <- torch::optim_adam(params_tensor, lr = 0.01, ...)
    } else {
      opt <- torch::optim_adam(params_list, lr = 0.01, ...)
    }
  } else {
    stop("Only 'lbfgs' and 'adam' optimizers are currently supported")
  }

  # Track iteration count and convergence
  iter_count <- 0L
  converged <- FALSE
  conv_message <- ""
  last_loss <- Inf

  # Optimization loop
  closure_fn <- function() {
    iter_count <<- iter_count + 1L
    opt$zero_grad()
    loss <- compute_loss()
    
    if (is_torch) {
      # Use autograd for torch-native functions
      loss$backward()
    }
    
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
  if (is_torch) {
    final_params <- as.numeric(params_tensor$detach()$cpu())
    names(final_params) <- param_names
  } else {
    final_params <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
    names(final_params) <- param_names
  }

  # Compute final log-likelihood
  if (is_torch) {
    final_ll <- -as.numeric(compute_loss()$item())
  } else {
    final_ll <- loglik(final_params, data)
  }

  # Extract gradients
  if (is_torch) {
    # Get gradients from autograd
    if (!is.null(params_tensor$grad)) {
      final_grad <- as.numeric(params_tensor$grad$detach()$cpu())
      names(final_grad) <- param_names
      grad_norm <- sqrt(sum(final_grad^2))
    } else {
      # Try to recompute gradients
      tryCatch({
        params_tensor$requires_grad_(TRUE)
        params_tensor$grad <- NULL
        ll_tensor <- do.call(loglik, c(list(params_tensor, data), dots))
        ll_tensor$backward()
        final_grad <- as.numeric(params_tensor$grad$detach()$cpu())
        names(final_grad) <- param_names
        grad_norm <- sqrt(sum(final_grad^2))
      }, error = function(e) {
        final_grad <<- rep(NA_real_, n_params)
        names(final_grad) <<- param_names
        grad_norm <<- NA_real_
      })
    }
  } else {
    # Recompute gradients using finite differences
    # For R functions, data is already R format, so we can use it directly
    final_grad <- -finite_diff_grad(loglik, final_params, data)
    names(final_grad) <- param_names
    grad_norm <- sqrt(sum(final_grad^2))
  }

  # Compute Hessian and vcov (information matrix)
  vcov_matrix <- NULL
  if (is_torch) {
    # For torch-native functions, compute Hessian using autograd
    tryCatch({
      # Re-enable gradients
      params_tensor$requires_grad_(TRUE)
      params_tensor$grad <- NULL
      
      # Compute log-likelihood
      ll_tensor <- loglik(params_tensor, data)
      
      # Compute first derivatives
      grad_list <- torch::autograd_grad(
        outputs = ll_tensor,
        inputs = params_tensor,
        create_graph = TRUE
      )[[1]]
      
      # Compute second derivatives (Hessian)
      hessian_list <- list()
      for (i in seq_len(n_params)) {
        hess_row <- torch::autograd_grad(
          outputs = grad_list[i],
          inputs = params_tensor,
          retain_graph = TRUE
        )[[1]]
        hessian_list[[i]] <- as.numeric(hess_row$detach()$cpu())
      }
      
      hessian <- do.call(rbind, hessian_list)
      # Information matrix is negative Hessian
      info_matrix <- -hessian
      
      # Compute vcov (inverse of information matrix)
      tryCatch({
        vcov_matrix <- solve(info_matrix)
        dimnames(vcov_matrix) <- list(param_names, param_names)
      }, error = function(e) {
        warning("Could not compute vcov: information matrix is singular")
      })
    }, error = function(e) {
      warning("Could not compute Hessian using autograd: ", conditionMessage(e))
    })
  } else {
    # For R functions, use finite differences for Hessian
    tryCatch({
      hessian <- finite_diff_hessian(loglik, final_params, data)
      # Information matrix is negative Hessian
      info_matrix <- -hessian
      
      # Compute vcov (inverse of information matrix)
      tryCatch({
        vcov_matrix <- solve(info_matrix)
        dimnames(vcov_matrix) <- list(param_names, param_names)
      }, error = function(e) {
        warning("Could not compute vcov: information matrix is singular")
      })
    }, error = function(e) {
      warning("Could not compute Hessian: ", conditionMessage(e))
    })
  }

  # Determine convergence code
  convergence_code <- if (converged && !is.na(grad_norm) && grad_norm < tolerance) {
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
    vcov = vcov_matrix,
    call = call
  )
}
