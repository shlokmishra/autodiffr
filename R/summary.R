#' Summary method for autodiffr_fit objects
#'
#' @param object An object of class `autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return An object of class `summary.autodiffr_fit` with detailed information
#'
#' @export
summary.autodiffr_fit <- function(object, ...) {
  # Create summary structure
  summary_obj <- list(
    fit = object,
    coefficients = object$coefficients,
    loglik = object$loglik,
    convergence = object$convergence,
    message = object$message,
    iterations = object$iterations,
    gradient_norm = object$gradient_norm,
    gradient = object$gradient,
    optimizer = object$optimizer
  )

  class(summary_obj) <- "summary.autodiffr_fit"
  return(summary_obj)
}

#' Print method for summary.autodiffr_fit objects
#'
#' @param x An object of class `summary.autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return Invisibly returns `x`
#'
#' @export
print.summary.autodiffr_fit <- function(x, ...) {
  cat("Autodiffr MLE Fit Summary\n")
  cat("==========================\n\n")

  # Print call if available
  if (!is.null(x$fit$call)) {
    cat("Call:\n")
    print(x$fit$call)
    cat("\n")
  }

  # Print coefficients
  if (length(x$coefficients) > 0) {
    cat("Coefficients:\n")
    print(x$coefficients)
    cat("\n")
  }

  # Print log-likelihood
  if (!is.na(x$loglik)) {
    cat("Log-likelihood:", x$loglik, "\n\n")
  }

  # Print optimization details
  cat("Optimization Details:\n")
  cat("  Optimizer:", x$optimizer, "\n")

  if (!is.na(x$iterations)) {
    cat("  Iterations:", x$iterations, "\n")
  }

  # Print convergence status
  cat("  Convergence:\n")
  if (x$convergence == 0L) {
    cat("    Status: Success\n")
  } else if (x$convergence == 1L) {
    cat("    Status: Converged (gradient may be large)\n")
  } else {
    cat("    Status: Did not converge\n")
  }

  if (length(x$message) > 0) {
    cat("    Message:", x$message, "\n")
  }

  # Print gradient information
  if (!is.na(x$gradient_norm)) {
    cat("  Gradient norm:", x$gradient_norm, "\n")
  }

  if (length(x$gradient) > 0 && !all(is.na(x$gradient))) {
    cat("\n  Gradients:\n")
    print(x$gradient)
  }

  invisible(x)
}


