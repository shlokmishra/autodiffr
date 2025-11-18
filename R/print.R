#' Print method for autodiffr_fit objects
#'
#' @param x An object of class `autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return Invisibly returns `x`
#'
#' @export
print.autodiffr_fit <- function(x, ...) {
  cat("Autodiffr MLE Fit\n")
  cat("==================\n\n")

  # Print coefficients
  if (length(x$coefficients) > 0) {
    cat("Coefficients:\n")
    print(x$coefficients)
    cat("\n")
  }

  # Print log-likelihood
  if (!is.na(x$loglik)) {
    cat("Log-likelihood:", x$loglik, "\n")
  }

  # Print convergence status
  cat("\nConvergence:\n")
  if (x$convergence == 0L) {
    cat("  Status: Success\n")
  } else if (x$convergence == 1L) {
    cat("  Status: Converged (gradient may be large)\n")
  } else {
    cat("  Status: Did not converge\n")
  }

  if (length(x$message) > 0) {
    cat("  Message:", x$message, "\n")
  }

  if (!is.na(x$iterations)) {
    cat("  Iterations:", x$iterations, "\n")
  }

  if (!is.na(x$gradient_norm)) {
    cat("  Gradient norm:", x$gradient_norm, "\n")
  }

  invisible(x)
}

