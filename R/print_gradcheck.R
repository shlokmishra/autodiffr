#' Print method for autodiffr_gradcheck objects
#'
#' @param x An object of class `autodiffr_gradcheck`
#' @param ... Additional arguments (currently ignored)
#'
#' @return Invisibly returns `x`
#'
#' @export
print.autodiffr_gradcheck <- function(x, ...) {
  cat("Gradient Check Results\n")
  cat("======================\n\n")
  
  cat("Gradients OK:", x$ok, "\n")
  cat("Maximum absolute error:", x$max_abs_err, "\n")
  cat("Maximum relative error:", x$max_rel_err, "\n")
  cat("Relative tolerance:", x$rel_tol, "\n\n")
  
  if (!x$ok) {
    cat("Parameters with largest relative errors:\n")
    # Sort by relative error and show worst offenders
    worst_idx <- order(x$rel_err, decreasing = TRUE)[seq_len(min(5, length(x$rel_err)))]
    for (i in worst_idx) {
      param_name <- names(x$rel_err)[i]
      cat(sprintf("  %s: rel_err = %.2e, abs_err = %.2e\n",
                  param_name, x$rel_err[i], x$abs_err[i]))
    }
    cat("\n")
  }
  
  cat("Gradient comparison:\n")
  comparison <- data.frame(
    Parameter = names(x$grad_autograd),
    Autograd = x$grad_autograd,
    FiniteDiff = x$grad_fd,
    AbsError = x$abs_err,
    RelError = x$rel_err,
    stringsAsFactors = FALSE
  )
  print(comparison, row.names = FALSE)
  
  invisible(x)
}

