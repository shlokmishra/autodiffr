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
#' @param constraints Optional constraint specification. Can be a single constraint
#'   object (from `positive()`, `simplex()`, etc.) or a `constraints()` list.
#'   When constraints are provided, parameters are transformed to unconstrained
#'   space for optimization, and the log-likelihood is adjusted by the log-Jacobian
#'   of the transformation. The final estimates are returned in the constrained space.
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
                       constraints = NULL,
                       ...) {
  # Store the call
  call <- match.call()

  # Validate inputs
  validate_loglik(loglik)
  validate_start(start)
  
  # Validate constraints if provided
  if (!is.null(constraints)) {
    if (!inherits(constraints, "autodiffr_constraint") && 
        !inherits(constraints, "autodiffr_constraints")) {
      stop("constraints must be a constraint object or constraints() list")
    }
  }

  # Check torch availability
  if (!requireNamespace("torch", quietly = TRUE)) {
    stop("torch package is required but not installed. ",
         "Install it with: install.packages('torch')")
  }

  # Handle constraints: transform start from constrained to unconstrained space
  param_names <- names(start)
  n_params <- length(start)
  
  if (!is.null(constraints)) {
    # Transform constrained start to unconstrained
    start_unconstrained <- constraint_to_unconstrained(start, constraints, param_names)
  } else {
    start_unconstrained <- start
  }

  # Create torch tensors for parameters (in unconstrained space)
  params_list <- lapply(seq_along(start_unconstrained), function(i) {
    torch::torch_tensor(
      start_unconstrained[i],
      requires_grad = TRUE,
      dtype = torch::torch_float64()
    )
  })
  names(params_list) <- param_names

  # Create a single tensor for torch-native functions (unconstrained)
  params_tensor <- torch::torch_tensor(
    as.numeric(start_unconstrained),
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
      # Transform unconstrained u to constrained theta
      if (!is.null(constraints)) {
        constraint_result <- apply_constraints(params_tensor, constraints, param_names)
        theta_constrained <- constraint_result$theta
        log_jacobian <- constraint_result$log_jacobian
      } else {
        theta_constrained <- params_tensor
        log_jacobian <- torch::torch_tensor(0.0, dtype = torch::torch_float64())
      }
      
      # Call user's torch-native function with constrained parameters
      ll_tensor <- loglik(theta_constrained, data)
      
      # Ensure it's a scalar tensor
      if (ll_tensor$numel() != 1) {
        stop("loglik must return a scalar tensor")
      }
      
      # Add log-Jacobian: loglik_u = loglik(theta) + log|J|
      ll_adjusted <- ll_tensor + log_jacobian
      
      # Return negative (we minimize)
      loss <- -ll_adjusted
      return(loss)
    }
  } else {
    # R function mode: use finite differences
    compute_loss <- function() {
      # Convert torch tensors to R numeric vector (unconstrained)
      u_r <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
      names(u_r) <- param_names

      # Transform to constrained space
      if (!is.null(constraints)) {
        constraint_result <- apply_constraints_r(u_r, constraints, param_names)
        theta_r <- constraint_result$theta
        log_jacobian <- constraint_result$log_jacobian
      } else {
        theta_r <- u_r
        log_jacobian <- 0.0
      }

      # Call user's R function with constrained parameters
      ll_value <- loglik(theta_r, data)
      
      # Add log-Jacobian
      ll_adjusted <- ll_value + log_jacobian

      # Convert to torch tensor (negative because we'll minimize)
      loss_tensor <- torch::torch_tensor(-ll_adjusted, dtype = torch::torch_float64(),
                                         requires_grad = TRUE)

      # Compute gradients using finite differences
      # We need to compute gradient w.r.t. unconstrained u
      # dL/du = dL/dtheta * dtheta/du = dL/dtheta * J
      # For now, compute dL/dtheta and approximate dtheta/du
      eps <- 1e-5
      grads_theta <- finite_diff_grad(function(theta, data) {
        if (!is.null(constraints)) {
          # Transform theta to get u, then back to theta (to ensure consistency)
          # Actually, we need to compute gradient w.r.t. u
          # Let's compute it numerically
          u_test <- constraint_to_unconstrained(theta, constraints, param_names)
          result <- apply_constraints_r(u_test, constraints, param_names)
          loglik(result$theta, data) + result$log_jacobian
        } else {
          loglik(theta, data)
        }
      }, theta_r, data, eps)
      
      # For constraints, we need to transform gradients from theta to u
      # For positive: dL/du = dL/dtheta * exp(u) = dL/dtheta * theta
      # For simplex: more complex, approximate numerically
      if (!is.null(constraints)) {
        grads_u <- numeric(length(u_r))
        for (i in seq_along(u_r)) {
          u_plus <- u_r
          u_plus[i] <- u_plus[i] + eps
          result_plus <- apply_constraints_r(u_plus, constraints, param_names)
          ll_plus <- loglik(result_plus$theta, data) + result_plus$log_jacobian
          grads_u[i] <- (ll_plus - ll_adjusted) / eps
        }
        grads <- grads_u
      } else {
        grads <- grads_theta
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

  # Extract final values (in unconstrained space)
  if (is_torch) {
    final_params_u <- as.numeric(params_tensor$detach()$cpu())
    names(final_params_u) <- param_names
  } else {
    final_params_u <- vapply(params_list, function(p) as.numeric(p$item()), numeric(1))
    names(final_params_u) <- param_names
  }
  
  # Transform back to constrained space for final results
  if (!is.null(constraints)) {
    if (is_torch) {
      # Convert to torch tensor, apply constraints, convert back
      u_tensor <- torch::torch_tensor(as.numeric(final_params_u), dtype = torch::torch_float64())
      constraint_result <- apply_constraints(u_tensor, constraints, param_names)
      final_params <- as.numeric(constraint_result$theta$detach()$cpu())
      names(final_params) <- param_names
    } else {
      constraint_result <- apply_constraints_r(final_params_u, constraints, param_names)
      final_params <- constraint_result$theta
      names(final_params) <- param_names
    }
  } else {
    final_params <- final_params_u
  }

  # Compute final log-likelihood (with constraints if applicable)
  if (is_torch) {
    final_ll <- -as.numeric(compute_loss()$item())
  } else {
    if (!is.null(constraints)) {
      constraint_result <- apply_constraints_r(final_params_u, constraints, param_names)
      final_ll <- loglik(constraint_result$theta, data) + constraint_result$log_jacobian
    } else {
      final_ll <- loglik(final_params, data)
    }
  }

  # Extract gradients (in unconstrained space)
  # Note: Gradients are in unconstrained space since that's what the optimizer uses
  if (is_torch) {
    # Get gradients from autograd (these are w.r.t. unconstrained u)
    if (!is.null(params_tensor$grad)) {
      final_grad_u <- as.numeric(params_tensor$grad$detach()$cpu())
      names(final_grad_u) <- param_names
      grad_norm <- sqrt(sum(final_grad_u^2))
    } else {
      # Try to recompute gradients
      tryCatch({
        params_tensor$requires_grad_(TRUE)
        params_tensor$grad <- NULL
        loss <- compute_loss()
        loss$backward()
        final_grad_u <- as.numeric(params_tensor$grad$detach()$cpu())
        names(final_grad_u) <- param_names
        grad_norm <- sqrt(sum(final_grad_u^2))
      }, error = function(e) {
        final_grad_u <<- rep(NA_real_, n_params)
        names(final_grad_u) <<- param_names
        grad_norm <<- NA_real_
      })
    }
    # For reporting, use unconstrained gradients
    final_grad <- final_grad_u
  } else {
    # Recompute gradients using finite differences (w.r.t. unconstrained u)
    if (!is.null(constraints)) {
      # Compute gradient w.r.t. unconstrained parameters
      eps <- 1e-5
      grads_u <- numeric(length(final_params_u))
      for (i in seq_along(final_params_u)) {
        u_plus <- final_params_u
        u_plus[i] <- u_plus[i] + eps
        result_plus <- apply_constraints_r(u_plus, constraints, param_names)
        ll_plus <- loglik(result_plus$theta, data) + result_plus$log_jacobian
        grads_u[i] <- (ll_plus - final_ll) / eps
      }
      final_grad <- -grads_u
    } else {
      final_grad <- -finite_diff_grad(loglik, final_params, data)
    }
    names(final_grad) <- param_names
    grad_norm <- sqrt(sum(final_grad^2))
  }

  # Compute Hessian and vcov (information matrix)
  # Note: When constraints are present, vcov is computed in unconstrained space
  vcov_matrix <- NULL
  if (is_torch) {
    # For torch-native functions, compute Hessian using autograd
    # Compute Hessian of the adjusted log-likelihood (including log-Jacobian)
    tryCatch({
      # Re-enable gradients
      params_tensor$requires_grad_(TRUE)
      params_tensor$grad <- NULL
      
      # Compute adjusted log-likelihood (with constraints if applicable)
      if (!is.null(constraints)) {
        constraint_result <- apply_constraints(params_tensor, constraints, param_names)
        theta_constrained <- constraint_result$theta
        log_jacobian <- constraint_result$log_jacobian
        ll_tensor <- loglik(theta_constrained, data) + log_jacobian
      } else {
        ll_tensor <- loglik(params_tensor, data)
      }
      
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
      # Compute Hessian of adjusted log-likelihood
      if (!is.null(constraints)) {
        # Compute Hessian w.r.t. unconstrained parameters
        # Use a wrapper function that includes constraints
        loglik_adjusted <- function(u, data) {
          result <- apply_constraints_r(u, constraints, param_names)
          loglik(result$theta, data) + result$log_jacobian
        }
        hessian <- finite_diff_hessian(loglik_adjusted, final_params_u, data)
      } else {
        hessian <- finite_diff_hessian(loglik, final_params, data)
      }
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
    method = "mle",
    call = call
  )
}
