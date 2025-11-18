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

