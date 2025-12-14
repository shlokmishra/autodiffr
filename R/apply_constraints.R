# Helper to transform constrained start values to unconstrained
# Inverse transformations for starting values
constraint_to_unconstrained <- function(theta_start, constraints_spec, param_names) {
  if (is.null(constraints_spec) || length(constraints_spec) == 0) {
    return(theta_start)
  }
  
  if (inherits(constraints_spec, "autodiffr_constraints")) {
    constraints_list <- constraints_spec
  } else if (inherits(constraints_spec, "autodiffr_constraint")) {
    constraints_list <- list(constraints_spec)
  } else {
    stop("constraints must be a constraint object or constraints() list")
  }
  
  u_start <- theta_start
  
  for (constr in constraints_list) {
    if (constr$type == "positive") {
      idx <- which(param_names == constr$name)
      if (length(idx) != 1) {
        stop("Parameter '", constr$name, "' not found or ambiguous")
      }
      if (theta_start[idx] <= 0) {
        stop("Starting value for '", constr$name, "' must be positive")
      }
      u_start[idx] <- log(theta_start[idx])
      
    } else if (constr$type == "simplex") {
      idx_start <- which(param_names == constr$name)
      if (length(idx_start) != 1) {
        stop("Parameter '", constr$name, "' not found or ambiguous")
      }
      idx_block <- idx_start:(idx_start + constr$k - 1)
      if (idx_start + constr$k - 1 > length(param_names)) {
        stop("Not enough parameters for simplex constraint")
      }
      
      theta_block <- theta_start[idx_block]
      if (any(theta_block < 0) || abs(sum(theta_block) - 1) > 1e-6) {
        stop("Starting values for simplex '", constr$name, "' must be non-negative and sum to 1")
      }
      
      # Inverse softmax: use log-odds (log(theta/theta_k))
      # For numerical stability, use log(theta) - mean(log(theta))
      log_theta <- log(pmax(theta_block, 1e-10))  # Avoid log(0)
      u_start[idx_block] <- log_theta - mean(log_theta)
      
    } else if (constr$type == "corr_matrix") {
      stop("corr_matrix constraints are not yet implemented")
    }
  }
  
  return(u_start)
}

# Internal function to apply constraints and compute log-Jacobian
# This maps unconstrained parameters u to constrained parameters theta

apply_constraints <- function(u, constraints_spec, param_names) {
  # u is a torch tensor of unconstrained parameters
  # constraints_spec is a list of constraint objects
  # param_names is the full list of parameter names
  
  if (is.null(constraints_spec) || length(constraints_spec) == 0) {
    # No constraints: theta = u, log_jacobian = 0
    return(list(
      theta = u,
      log_jacobian = torch::torch_tensor(0.0, dtype = torch::torch_float64())
    ))
  }
  
  # Convert constraints_spec to list if it's a constraints object
  if (inherits(constraints_spec, "autodiffr_constraints")) {
    constraints_list <- constraints_spec
  } else if (inherits(constraints_spec, "autodiffr_constraint")) {
    constraints_list <- list(constraints_spec)
  } else {
    stop("constraints must be a constraint object or constraints() list")
  }
  
  # Start with theta = u (copy)
  theta <- u$clone()
  log_jacobian <- torch::torch_tensor(0.0, dtype = torch::torch_float64())
  
  # Track which parameters have been constrained
  constrained_params <- character(0)
  
  # Apply each constraint
  for (constr in constraints_list) {
    if (constr$type == "positive") {
      # Find indices for this parameter
      idx <- which(param_names == constr$name)
      if (length(idx) == 0) {
        stop("Parameter '", constr$name, "' not found in parameter names")
      }
      if (length(idx) > 1) {
        stop("Parameter '", constr$name, "' appears multiple times")
      }
      if (constr$name %in% constrained_params) {
        stop("Parameter '", constr$name, "' has multiple constraints")
      }
      
      # Apply exponential transformation
      theta[idx] <- torch::torch_exp(u[idx])
      
      # Log-Jacobian: for exp(u), J = exp(u), so log|J| = u
      log_jacobian <- log_jacobian + u[idx]
      
      constrained_params <- c(constrained_params, constr$name)
      
    } else if (constr$type == "simplex") {
      # Find indices for this parameter block
      # For simplex, the parameter name should match the first element
      # and we expect k consecutive parameters
      idx_start <- which(param_names == constr$name)
      if (length(idx_start) == 0) {
        stop("Parameter '", constr$name, "' not found in parameter names")
      }
      if (length(idx_start) > 1) {
        stop("Parameter '", constr$name, "' appears multiple times")
      }
      
      # Check if we have enough parameters
      if (idx_start + constr$k - 1 > length(param_names)) {
        stop("Not enough parameters for simplex constraint on '", constr$name, "'")
      }
      
      # Get the block of parameters
      idx_block <- idx_start:(idx_start + constr$k - 1)
      
      # Check if any are already constrained
      if (any(param_names[idx_block] %in% constrained_params)) {
        stop("Some parameters in simplex '", constr$name, "' are already constrained")
      }
      
      # Apply softmax transformation
      u_block <- u[idx_block]
      theta_block <- torch::nnf_softmax(u_block, dim = 0L)
      theta[idx_block] <- theta_block
      
      # Log-Jacobian: for softmax, this is more complex
      # For v0.1, we set it to 0 and document that it's not yet implemented
      # TODO: Implement proper log-Jacobian for softmax
      log_jacobian <- log_jacobian + torch::torch_tensor(0.0, dtype = torch::torch_float64())
      
      constrained_params <- c(constrained_params, param_names[idx_block])
      
    } else if (constr$type == "corr_matrix") {
      stop("corr_matrix constraints are not yet implemented")
    } else {
      stop("Unknown constraint type: ", constr$type)
    }
  }
  
  return(list(
    theta = theta,
    log_jacobian = log_jacobian
  ))
}

# Helper to apply constraints to R numeric vector (for R function mode)
apply_constraints_r <- function(u_r, constraints_spec, param_names) {
  # u_r is a numeric vector of unconstrained parameters
  # Returns list(theta = theta_r, log_jacobian = logJ_r)
  
  if (is.null(constraints_spec) || length(constraints_spec) == 0) {
    return(list(
      theta = u_r,
      log_jacobian = 0.0
    ))
  }
  
  # Convert constraints_spec to list if needed
  if (inherits(constraints_spec, "autodiffr_constraints")) {
    constraints_list <- constraints_spec
  } else if (inherits(constraints_spec, "autodiffr_constraint")) {
    constraints_list <- list(constraints_spec)
  } else {
    stop("constraints must be a constraint object or constraints() list")
  }
  
  theta_r <- u_r
  log_jacobian <- 0.0
  constrained_params <- character(0)
  
  for (constr in constraints_list) {
    if (constr$type == "positive") {
      idx <- which(param_names == constr$name)
      if (length(idx) != 1) {
        stop("Parameter '", constr$name, "' not found or ambiguous")
      }
      if (constr$name %in% constrained_params) {
        stop("Parameter '", constr$name, "' has multiple constraints")
      }
      
      theta_r[idx] <- exp(u_r[idx])
      log_jacobian <- log_jacobian + u_r[idx]
      constrained_params <- c(constrained_params, constr$name)
      
    } else if (constr$type == "simplex") {
      idx_start <- which(param_names == constr$name)
      if (length(idx_start) != 1) {
        stop("Parameter '", constr$name, "' not found or ambiguous")
      }
      
      idx_block <- idx_start:(idx_start + constr$k - 1)
      if (idx_start + constr$k - 1 > length(param_names)) {
        stop("Not enough parameters for simplex constraint")
      }
      
      if (any(param_names[idx_block] %in% constrained_params)) {
        stop("Some parameters in simplex are already constrained")
      }
      
      # Apply softmax
      u_block <- u_r[idx_block]
      exp_u <- exp(u_block - max(u_block))  # Numerical stability
      theta_block <- exp_u / sum(exp_u)
      theta_r[idx_block] <- theta_block
      
      # Log-Jacobian: set to 0 for v0.1
      log_jacobian <- log_jacobian + 0.0
      
      constrained_params <- c(constrained_params, param_names[idx_block])
      
    } else if (constr$type == "corr_matrix") {
      stop("corr_matrix constraints are not yet implemented")
    } else {
      stop("Unknown constraint type: ", constr$type)
    }
  }
  
  return(list(
    theta = theta_r,
    log_jacobian = log_jacobian
  ))
}

