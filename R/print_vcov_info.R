#' Print method for autodiffr_vcov_info objects
#'
#' @param x An object of class `autodiffr_vcov_info`
#' @param ... Additional arguments (currently ignored)
#'
#' @return Invisibly returns `x`
#'
#' @export
print.autodiffr_vcov_info <- function(x, ...) {
  cat("Variance-Covariance Matrix Information\n")
  cat("=======================================\n\n")

  cat("Type:", x$type, "\n")
  cat("Method:", x$method, "\n")

  if (!is.na(x$nobs)) {
    cat("Number of observations:", x$nobs, "\n")
  }

  cat("\nCondition Number:", x$cond_number, "\n")

  # Determine if well-conditioned
  # Condition number < 1e12 is generally considered acceptable for double precision
  well_conditioned <- x$cond_number < 1e12
  cat("Well-conditioned:", well_conditioned, "\n")

  if (length(x$eigenvalues) > 0) {
    cat("\nInformation Matrix Eigenvalues:\n")
    cat("  Minimum:", min(x$eigenvalues), "\n")
    cat("  Maximum:", max(x$eigenvalues), "\n")
    cat("  Range:", max(x$eigenvalues) - min(x$eigenvalues), "\n")

    # Check for negative eigenvalues (indicates non-positive definite)
    if (any(x$eigenvalues < 0)) {
      cat("\n  Warning: Information matrix has negative eigenvalues!\n")
      cat("  This suggests the matrix is not positive definite.\n")
    }
  }

  cat("\nVariance-Covariance Matrix:\n")
  print(x$vcov)

  invisible(x)
}

