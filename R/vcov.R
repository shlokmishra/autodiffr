#' Extract Variance-Covariance Matrix from autodiffr_fit
#'
#' @param object An object of class `autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return A variance-covariance matrix, or NULL if not available
#'
#' @export
vcov.autodiffr_fit <- function(object, ...) {
  object$vcov
}


