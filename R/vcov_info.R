#' Extract Variance-Covariance Matrix with Diagnostics
#'
#' Computes and returns the variance-covariance matrix along with diagnostic
#' information such as eigenvalues, condition number, and information matrix.
#'
#' @param object An object of class `autodiffr_fit`
#' @param type Character string specifying the type of variance-covariance matrix.
#'   Options: `"observed"` (default), `"expected"`, or `"sandwich"`.
#' @param ... Additional arguments (currently ignored)
#'
#' @return An object of class `autodiffr_vcov_info` containing:
#'   \item{vcov}{Variance-covariance matrix}
#'   \item{info}{Information matrix (negative Hessian)}
#'   \item{eigenvalues}{Eigenvalues of the information matrix}
#'   \item{cond_number}{Condition number of the information matrix}
#'   \item{type}{Type of variance-covariance matrix requested}
#'   \item{method}{Estimation method ("mle" or "mest")}
#'   \item{nobs}{Number of observations (if available)}
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(torch)
#' set.seed(123)
#' data <- rnorm(100, mean = 5, sd = 2)
#'
#' loglik_r <- function(theta, data) {
#'   mu <- theta["mu"]
#'   sigma <- theta["sigma"]
#'   sum(dnorm(data, mean = mu, sd = sigma, log = TRUE))
#' }
#'
#' start <- c(mu = 0, sigma = 1)
#' fit <- optim_mle(loglik_r, start, data)
#'
#' vcov_info(fit)
#' vcov_info(fit, type = "observed")
#' }
vcov_info <- function(object, ...) {
  UseMethod("vcov_info")
}

#' @rdname vcov_info
#' @export
vcov_info.autodiffr_fit <- function(
  object,
  type = c("observed", "expected", "sandwich"),
  ...
) {
  type <- match.arg(type)

  # For now, only "observed" is fully implemented
  if (type == "expected") {
    stop("Expected information matrix not yet implemented. ",
         "Use type = 'observed' for now.")
  }

  if (type == "sandwich") {
    stop("Sandwich variance-covariance matrix is only available for ",
         "M-estimation fits (optim_mest). Use type = 'observed' for MLE fits.")
  }

  # Get vcov from fit object
  vcov_mat <- object$vcov

  if (is.null(vcov_mat)) {
    stop("Variance-covariance matrix not available in fit object. ",
         "It may not have been computed during optimization.")
  }

  # Compute information matrix (inverse of vcov)
  info_mat <- tryCatch({
    solve(vcov_mat)
  }, error = function(e) {
    stop("Could not compute information matrix: variance-covariance matrix is singular")
  })

  # Compute eigenvalues
  eigen_vals <- eigen(info_mat, only.values = TRUE)$values

  # Compute condition number
  cond_num <- kappa(info_mat)

  # Determine method (for now, all fits are MLE)
  method <- "mle"

  # Get number of observations (not stored yet, so NA)
  nobs <- NA_real_

  # Create result object
  result <- list(
    vcov = vcov_mat,
    info = info_mat,
    eigenvalues = eigen_vals,
    cond_number = cond_num,
    type = type,
    method = method,
    nobs = nobs
  )

  class(result) <- "autodiffr_vcov_info"
  return(result)
}

