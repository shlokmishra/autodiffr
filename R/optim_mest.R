#' M-Estimation using Automatic Differentiation
#'
#' Fit M-estimators by solving estimating equations using automatic differentiation.
#' This function minimizes the sum of squared mean estimating equations and computes
#' sandwich (Godambe) variance-covariance matrices.
#'
#' @param psi A function that computes per-observation estimating equations. Must
#'   accept at least two arguments: `theta` (parameters) and `data`. In torch-native
#'   mode, it should return a tensor of shape (n, p) where n is the number of
#'   observations and p is the number of parameters. In R function mode, it should
#'   return a matrix with n rows and p columns. Each row represents the contribution
#'   from one observation, and the column means should be zero at the solution.
#' @param theta0 A named numeric vector of starting values for the parameters.
#' @param data The data object to be passed to `psi`.
#' @param ... Additional arguments passed to `psi`.
#' @param method Character string specifying the optimizer to use. Options:
#'   `"lbfgs"` (default) or `"adam"`.
#' @param use_autodiff Logical. If `TRUE` (default), automatically detect
#'   torch-native functions and use autograd when possible. If `FALSE`, use
#'   finite-difference gradients.
#' @param control List of control parameters for optimization (max_iter, tolerance, etc.).
#' @param constraints Optional constraint specification. Can be a single constraint
#'   object or a `constraints()` list. See `optim_mle()` for details.
#'
#' @return An object of class `autodiffr_fit` containing:
#'   \item{coefficients}{Named numeric vector of parameter estimates}
#'   \item{convergence}{Convergence code (0 = success)}
#'   \item{message}{Convergence message}
#'   \item{iterations}{Number of iterations}
#'   \item{gradient_norm}{Norm of the final gradient}
#'   \item{gradient}{Named numeric vector of final gradients}
#'   \item{vcov}{Sandwich variance-covariance matrix}
#'   \item{method}{Estimation method ("mest")}
#'   \item{call}{The original function call}
#'
#' @details
#' M-estimation solves estimating equations of the form:
#' \deqn{\sum_{i=1}^n \psi_i(\theta) = 0}
#'
#' This function minimizes \eqn{Q(\theta) = 0.5 \sum (\bar{\psi}(\theta))^2} where
#' \eqn{\bar{\psi}(\theta)} is the mean of the per-observation estimating equations.
#'
#' The sandwich variance-covariance matrix is computed as:
#' \deqn{V = J^{-1} S (J^{-1})^T / n}
#' where \eqn{J = \partial \bar{\psi} / \partial \theta^T} (bread) and
#' \eqn{S = \text{cov}(\psi_i)} (meat).
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Example: Linear regression via M-estimation
#' library(torch)
#' set.seed(123)
#' n <- 100
#' X <- cbind(1, rnorm(n))
#' y <- X %*% c(2, 3) + rnorm(n)
#' data_list <- list(X = X, y = y)
#'
#' # Estimating equations for OLS
#' psi_ols <- function(theta, data) {
#'   beta <- theta[1:2]
#'   residuals <- data$y - data$X %*% beta
#'   psi_mat <- residuals * data$X
#'   return(psi_mat)
#' }
#'
#' start <- c(beta0 = 0, beta1 = 0)
#' fit <- optim_mest(psi_ols, start, data_list)
#' print(fit)
#' }
optim_mest <- function(psi,
                        theta0,
                        data,
                        ...,
                        method = c("lbfgs", "adam"),
                        use_autodiff = TRUE,
                        control = list(),
                        constraints = NULL) {
  # Store the call
  call <- match.call()

  # Validate inputs
  if (!is.function(psi)) {
    stop("psi must be a function")
  }
  if (length(formals(psi)) < 2) {
    stop("psi must accept at least two arguments: theta and data")
  }
  validate_start(theta0)
  
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

  # Match method
  method <- match.arg(method)

  # Extract control parameters
  max_iter <- if (!is.null(control$max_iter)) control$max_iter else 1000
  tolerance <- if (!is.null(control$tolerance)) control$tolerance else 1e-6
  small_sample <- if (!is.null(control$small_sample)) control$small_sample else FALSE

  # Handle constraints: transform start from constrained to unconstrained space
  param_names <- names(theta0)
  n_params <- length(theta0)
  
  if (!is.null(constraints)) {
    start_unconstrained <- constraint_to_unconstrained(theta0, constraints, param_names)
  } else {
    start_unconstrained <- theta0
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

  # Detect if function is torch-native
  is_torch <- FALSE
  if (use_autodiff) {
    # Test with a torch tensor matching the number of parameters
    test_tensor <- torch::torch_tensor(
      rep(0.0, n_params),
      requires_grad = TRUE,
      dtype = torch::torch_float64()
    )
    tryCatch({
      result <- psi(test_tensor, data, ...)
      is_torch <- inherits(result, "torch_tensor")
    }, error = function(e) {
      is_torch <- FALSE
    })
  }

  # Capture dots for use in closures
  dots <- list(...)

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
      
      # Call user's psi function
      psi_mat <- do.call(psi, c(list(theta_constrained, data), dots))
      
      # Check dimensions
      if (length(psi_mat$shape) != 2) {
        stop("psi must return a 2-dimensional tensor (n x p)")
      }
      
      # Compute mean estimating equations (mean across rows, i.e., dim=1 in 1-based indexing)
      psi_mean <- torch::torch_mean(psi_mat, dim = 1L)
      
      # Objective: Q = 0.5 * sum(psi_mean^2)
      Q <- 0.5 * torch::torch_sum(psi_mean^2)
      
      # Add log-Jacobian (though for M-estimation this is less critical)
      # For now, we'll include it for consistency
      Q_adjusted <- Q + log_jacobian
      
      return(Q_adjusted)
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

      # Call user's R function
      psi_mat <- do.call(psi, c(list(theta_r, data), dots))
      
      # Check dimensions
      if (!is.matrix(psi_mat) && !is.data.frame(psi_mat)) {
        stop("psi must return a matrix or data.frame with n rows and p columns")
      }
      psi_mat <- as.matrix(psi_mat)
      
      # Check that dimensions match
      if (ncol(psi_mat) != n_params) {
        stop("psi must return a matrix with ", n_params, " columns (one per parameter), ",
             "but got ", ncol(psi_mat), " columns")
      }
      
      # Compute mean estimating equations
      psi_mean <- colMeans(psi_mat)
      
      # Objective: Q = 0.5 * sum(psi_mean^2)
      Q <- 0.5 * sum(psi_mean^2)
      
      # Add log-Jacobian
      Q_adjusted <- Q + log_jacobian

      # For R functions, compute gradients using finite differences
      # and make loss depend on parameters so optimizers can work
      eps <- 1e-5
      grads_u <- numeric(length(u_r))
      for (i in seq_along(u_r)) {
        u_plus <- u_r
        u_plus[i] <- u_plus[i] + eps
        result_plus <- apply_constraints_r(u_plus, constraints, param_names)
        psi_mat_plus <- do.call(psi, c(list(result_plus$theta, data), dots))
        psi_mat_plus <- as.matrix(psi_mat_plus)
        psi_mean_plus <- colMeans(psi_mat_plus)
        Q_plus <- 0.5 * sum(psi_mean_plus^2) + result_plus$log_jacobian
        grads_u[i] <- (Q_plus - Q_adjusted) / eps
      }

      # Create a loss that depends on parameters via a linear combination
      # This allows us to manually set gradients that the optimizer will use
      # loss = Q + sum(params * grad_coeffs) where grad_coeffs are chosen to give desired gradients
      # Actually, simpler: make loss = sum(params * grads) + constant
      # But we need the loss value to be Q_adjusted
      # So: loss = Q_adjusted + sum(params * (grads - grads)) = Q_adjusted
      # Better: loss = sum(params * grads) + (Q_adjusted - sum(params * grads))
      grad_sum <- sum(u_r * grads_u)
      constant_part <- Q_adjusted - grad_sum
      
      # Build loss as linear combination of parameters
      loss_parts <- list()
      for (i in seq_along(params_list)) {
        loss_parts[[i]] <- params_list[[i]] * torch::torch_tensor(grads_u[i], dtype = torch::torch_float64())
      }
      loss_tensor <- Reduce(`+`, loss_parts) + torch::torch_tensor(constant_part, dtype = torch::torch_float64())
      
      # The gradients will be automatically computed as grads_u via autograd
      # But we've set it up so that d(loss)/d(params[i]) = grads_u[i]

      return(loss_tensor)
    }
  }

  # Set up optimizer
  if (method == "lbfgs") {
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
  } else if (method == "adam") {
    if (is_torch) {
      opt <- torch::optim_adam(params_tensor, lr = 0.01, ...)
    } else {
      opt <- torch::optim_adam(params_list, lr = 0.01, ...)
    }
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
    if (method == "lbfgs") {
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

  # Compute final objective value
  if (is_torch) {
    final_Q <- as.numeric(compute_loss()$item())
  } else {
    if (!is.null(constraints)) {
      constraint_result <- apply_constraints_r(final_params_u, constraints, param_names)
      theta_final <- constraint_result$theta
    } else {
      theta_final <- final_params
    }
    psi_mat_final <- do.call(psi, c(list(theta_final, data), dots))
    psi_mat_final <- as.matrix(psi_mat_final)
    psi_mean_final <- colMeans(psi_mat_final)
    final_Q <- 0.5 * sum(psi_mean_final^2)
  }

  # Extract gradients (in unconstrained space)
  if (is_torch) {
    if (!is.null(params_tensor$grad)) {
      final_grad_u <- as.numeric(params_tensor$grad$detach()$cpu())
      names(final_grad_u) <- param_names
      grad_norm <- sqrt(sum(final_grad_u^2))
    } else {
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
    final_grad <- final_grad_u
  } else {
    # Recompute gradients using finite differences
    eps <- 1e-5
    grads_u <- numeric(length(final_params_u))
    for (i in seq_along(final_params_u)) {
      u_plus <- final_params_u
      u_plus[i] <- u_plus[i] + eps
      result_plus <- apply_constraints_r(u_plus, constraints, param_names)
      psi_mat_plus <- do.call(psi, c(list(result_plus$theta, data), dots))
      psi_mat_plus <- as.matrix(psi_mat_plus)
      psi_mean_plus <- colMeans(psi_mat_plus)
      Q_plus <- 0.5 * sum(psi_mean_plus^2) + result_plus$log_jacobian
      grads_u[i] <- (Q_plus - final_Q) / eps
    }
    final_grad <- grads_u
    names(final_grad) <- param_names
    grad_norm <- sqrt(sum(final_grad^2))
  }

  # Compute sandwich variance-covariance matrix
  vcov_matrix <- NULL
  A_matrix <- NULL  # Bread matrix
  B_matrix <- NULL  # Meat matrix
  
  tryCatch({
    # Get final constrained parameters
    if (!is.null(constraints)) {
      if (is_torch) {
        u_tensor <- torch::torch_tensor(as.numeric(final_params_u), 
                                        requires_grad = TRUE,
                                        dtype = torch::torch_float64())
        constraint_result <- apply_constraints(u_tensor, constraints, param_names)
        theta_final_torch <- constraint_result$theta
      } else {
        constraint_result <- apply_constraints_r(final_params_u, constraints, param_names)
        theta_final_torch <- torch::torch_tensor(as.numeric(constraint_result$theta),
                                                 requires_grad = TRUE,
                                                 dtype = torch::torch_float64())
      }
    } else {
      if (is_torch) {
        theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                  requires_grad = TRUE,
                                                  dtype = torch::torch_float64())
      } else {
        theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                  requires_grad = TRUE,
                                                  dtype = torch::torch_float64())
      }
    }
    
    # Compute psi matrix at solution
    # For torch-native functions, we need to ensure theta_final_torch is properly set
    if (is_torch) {
      # Recompute with proper gradient tracking
      theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                requires_grad = TRUE,
                                                dtype = torch::torch_float64())
      if (!is.null(constraints)) {
        constraint_result <- apply_constraints(theta_final_torch, constraints, param_names)
        theta_final_torch <- constraint_result$theta
      }
    }
    
    psi_mat_soln <- do.call(psi, c(list(theta_final_torch, data), dots))
    if (inherits(psi_mat_soln, "torch_tensor")) {
      psi_mat_soln_r <- as.matrix(psi_mat_soln$detach()$cpu())
      n_obs <- nrow(psi_mat_soln_r)
      n_params_psi <- ncol(psi_mat_soln_r)
    } else {
      psi_mat_soln_r <- as.matrix(psi_mat_soln)
      n_obs <- nrow(psi_mat_soln_r)
      n_params_psi <- ncol(psi_mat_soln_r)
    }
    
    # Check dimensions match
    if (n_params_psi != n_params) {
      warning("Number of columns in psi matrix (", n_params_psi, 
              ") does not match number of parameters (", n_params, ")")
    }
    
    # Compute B (meat): sample covariance of psi_i
      B_matrix <- stats::cov(psi_mat_soln_r)
    
    # Compute A (bread): Jacobian of psi_mean w.r.t. theta
    # A[i,j] = d(psi_mean[j]) / d(theta[i])
    if (is_torch) {
      # Use autograd to compute Jacobian
      # Ensure theta_final_torch has gradient tracking
      if (!is.null(constraints)) {
        theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                  requires_grad = TRUE,
                                                  dtype = torch::torch_float64())
        constraint_result <- apply_constraints(theta_final_torch, constraints, param_names)
        theta_final_torch <- constraint_result$theta
      } else {
        theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                  requires_grad = TRUE,
                                                  dtype = torch::torch_float64())
      }
      
      # Compute Jacobian: d(psi_mean) / d(theta)
      # A[i,j] = d(psi_mean[j]) / d(theta[i])
      # We'll compute this by backpropagating each element of psi_mean separately
      A_list <- list()
      for (j in seq_len(n_params)) {
        # Recompute psi with fresh gradient tracking for each column
        # Need to ensure we're working with the constrained parameters
        if (!is.null(constraints)) {
          theta_unconstrained <- torch::torch_tensor(as.numeric(final_params_u),
                                                     requires_grad = TRUE,
                                                     dtype = torch::torch_float64())
          constraint_result <- apply_constraints(theta_unconstrained, constraints, param_names)
          theta_final_torch <- constraint_result$theta
        } else {
          theta_final_torch <- torch::torch_tensor(as.numeric(final_params),
                                                    requires_grad = TRUE,
                                                    dtype = torch::torch_float64())
        }
        
        psi_mat_grad <- do.call(psi, c(list(theta_final_torch, data), dots))
        # Extract j-th column and compute mean
        # Handle both torch tensors and R matrices
        if (inherits(psi_mat_grad, "torch_tensor")) {
          if (length(psi_mat_grad$shape) == 2) {
            # 2D tensor: extract column j (1-based indexing)
            tryCatch({
              psi_col_j <- psi_mat_grad[, j]
              psi_mean_j <- torch::torch_mean(psi_col_j)
            }, error = function(e) {
              # Fallback: convert to R and compute
              psi_mat_grad_r <- as.matrix(psi_mat_grad$detach()$cpu())
              psi_mean_j <<- torch::torch_tensor(mean(psi_mat_grad_r[, j]), 
                                                dtype = torch::torch_float64())
            })
          } else {
            # Not 2D, convert to R
            psi_mat_grad_r <- as.matrix(psi_mat_grad$detach()$cpu())
            psi_mean_j <- torch::torch_tensor(mean(psi_mat_grad_r[, j]), 
                                              dtype = torch::torch_float64())
          }
        } else {
          # R matrix, convert to torch
          psi_mat_grad_r <- as.matrix(psi_mat_grad)
          psi_mean_j <- torch::torch_tensor(mean(psi_mat_grad_r[, j]), 
                                            dtype = torch::torch_float64())
        }
        
        # Compute gradient of psi_mean_j w.r.t. theta_final_torch
        grad_j <- torch::autograd_grad(
          outputs = psi_mean_j,
          inputs = theta_final_torch,
          retain_graph = FALSE,
          create_graph = FALSE
        )[[1]]
        A_list[[j]] <- as.numeric(grad_j$detach()$cpu())
      }
      A_matrix <- do.call(rbind, A_list)  # p x p matrix
    } else {
      # Use finite differences for Jacobian
      eps <- 1e-5
      A_list <- list()
      psi_mean_base <- colMeans(psi_mat_soln_r)
      
      for (i in seq_len(n_params)) {
        theta_plus <- final_params
        theta_plus[i] <- theta_plus[i] + eps
        psi_mat_plus <- do.call(psi, c(list(theta_plus, data), dots))
        psi_mat_plus <- as.matrix(psi_mat_plus)
        psi_mean_plus <- colMeans(psi_mat_plus)
        A_list[[i]] <- (psi_mean_plus - psi_mean_base) / eps
      }
      A_matrix <- do.call(rbind, A_list)  # p x p matrix
    }
    
    # Compute sandwich variance: J_inv %*% S %*% t(J_inv) / n
    # where J = A (bread) and S = B (meat)
    tryCatch({
      J_inv <- solve(A_matrix)
      vcov_matrix <- J_inv %*% B_matrix %*% t(J_inv) / n_obs
      
      # Small-sample correction
      if (small_sample && n_obs > n_params) {
        correction <- n_obs / (n_obs - n_params)
        vcov_matrix <- vcov_matrix * correction
      }
      
      dimnames(vcov_matrix) <- list(param_names, param_names)
    }, error = function(e) {
      warning("Could not compute sandwich variance: ", conditionMessage(e))
    })
  }, error = function(e) {
    warning("Could not compute sandwich variance: ", conditionMessage(e))
  })

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
    loglik = -final_Q,  # Store negative Q as "loglik" for consistency
    convergence = convergence_code,
    message = conv_message,
    iterations = iter_count,
    gradient_norm = grad_norm,
    gradient = final_grad,
    optimizer = method,
    vcov = vcov_matrix,
    method = "mest",
    call = call
  )
}

