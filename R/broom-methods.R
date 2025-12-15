#' Tidy method for autodiffr_fit objects
#'
#' Returns a tibble with parameter estimates, standard errors, test statistics,
#' and p-values. This method works with the `broom` package for tidy model output.
#'
#' @param x An object of class `autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return A tibble with columns:
#'   \item{term}{Parameter name}
#'   \item{estimate}{Parameter estimate}
#'   \item{std.error}{Standard error}
#'   \item{statistic}{Z-statistic (estimate / std.error)}
#'   \item{p.value}{Two-sided p-value}
#'
#' @details
#' This function requires the `broom` and `tibble` packages. The `broom` package
#' is in Suggests, so it may not be installed. This method will work even if
#' `broom` is not attached, as it uses S3 method dispatch.
#'
#' @method tidy autodiffr_fit
#' @noRd
#'
#' @examples
#' \dontrun{
#' library(broom)
#' library(torch)
#' 
#' # Fit a model
#' fit <- optim_mle(loglik, start, data)
#' 
#' # Get tidy output
#' tidy(fit)
#' }
tidy.autodiffr_fit <- function(x, ...) {
  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("tibble package is required for tidy() method. ",
         "Install it with: install.packages('tibble')")
  }
  
  est <- stats::coef(x)
  vcov_mat <- stats::vcov(x)
  
  # Compute standard errors
  if (!is.null(vcov_mat)) {
    se <- sqrt(diag(vcov_mat))
  } else {
    se <- rep(NA_real_, length(est))
  }
  
  # Parameter names
  term <- names(est)
  if (is.null(term) || any(term == "")) {
    term <- paste0("par", seq_along(est))
  }
  
  # Compute statistics and p-values
  statistic <- est / se
  p.value <- 2 * stats::pnorm(-abs(statistic))
  
  tibble::tibble(
    term = term,
    estimate = est,
    std.error = se,
    statistic = statistic,
    p.value = p.value
  )
}

#' Glance method for autodiffr_fit objects
#'
#' Returns a one-row tibble with model-level statistics including log-likelihood,
#' AIC, BIC, convergence status, and method.
#'
#' @param x An object of class `autodiffr_fit`
#' @param ... Additional arguments (currently ignored)
#'
#' @return A one-row tibble with columns:
#'   \item{logLik}{Log-likelihood value}
#'   \item{AIC}{Akaike Information Criterion}
#'   \item{BIC}{Bayesian Information Criterion}
#'   \item{nobs}{Number of observations (if available)}
#'   \item{converged}{Logical, whether optimization converged}
#'   \item{method}{Estimation method ("mle" or "mest")}
#'   \item{iterations}{Number of optimization iterations}
#'
#' @details
#' This function requires the `broom` and `tibble` packages. AIC and BIC are
#' computed from the log-likelihood and number of parameters. The number of
#' observations (nobs) is not currently stored in fit objects and will be NA.
#'
#' @method glance autodiffr_fit
#' @noRd
#'
#' @examples
#' \dontrun{
#' library(broom)
#' 
#' fit <- optim_mle(loglik, start, data)
#' glance(fit)
#' }
glance.autodiffr_fit <- function(x, ...) {
  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("tibble package is required for glance() method. ",
         "Install it with: install.packages('tibble')")
  }
  
  logLik_val <- x$loglik
  n_params <- length(x$coefficients)
  
  # Compute AIC and BIC
  # AIC = -2 * logLik + 2 * k
  # BIC = -2 * logLik + k * log(n)
  # For BIC, we need nobs, which we don't currently store
  AIC_val <- if (!is.na(logLik_val)) {
    -2 * logLik_val + 2 * n_params
  } else {
    NA_real_
  }
  
  BIC_val <- NA_real_  # Need nobs for BIC
  
  # Convergence status
  converged <- x$convergence == 0L
  
  tibble::tibble(
    logLik = logLik_val,
    AIC = AIC_val,
    BIC = BIC_val,
    nobs = NA_integer_,
    converged = converged,
    method = x$method,
    iterations = x$iterations
  )
}

#' Augment method for autodiffr_fit objects
#'
#' Returns a tibble with the original data augmented with fitted values and
#' residuals. This is a minimal implementation that returns NA for fitted values
#' and residuals, as these require model-specific knowledge.
#'
#' @param x An object of class `autodiffr_fit`
#' @param data The original data (optional)
#' @param ... Additional arguments (currently ignored)
#'
#' @return A tibble with columns:
#'   \item{.fitted}{Fitted values (currently NA)}
#'   \item{.resid}{Residuals (currently NA)}
#'
#' @details
#' This is pretty minimal right now. To do it properly, we'd need model-specific
#' methods that know how to compute fitted values and residuals, which depends
#' on what kind of model you're fitting.
#'
#' @method augment autodiffr_fit
#' @noRd
#'
#' @examples
#' \dontrun{
#' library(broom)
#' 
#' fit <- optim_mle(loglik, start, data)
#' augment(fit, data = original_data)
#' }
augment.autodiffr_fit <- function(x, data = NULL, ...) {
  if (!requireNamespace("tibble", quietly = TRUE)) {
    stop("tibble package is required for augment() method. ",
         "Install it with: install.packages('tibble')")
  }
  
  if (is.null(data)) {
    warning("No data provided. Returning minimal augment output with NA values.")
    return(tibble::tibble(
      .fitted = NA_real_,
      .resid = NA_real_
    ))
  }
  
  # For now, return a minimal version
  # In the future, this could be extended to compute fitted values and residuals
  # based on the model type and stored information
  n_obs <- if (is.data.frame(data) || is.matrix(data)) {
    nrow(data)
  } else if (is.list(data) && length(data) > 0) {
    # Try to infer from first element
    if (is.vector(data[[1]]) || is.matrix(data[[1]])) {
      if (is.vector(data[[1]])) {
        length(data[[1]])
      } else if (is.matrix(data[[1]])) {
        nrow(data[[1]])
      } else {
        NA_integer_
      }
    } else {
      NA_integer_
    }
  } else {
    NA_integer_
  }
  
  if (is.na(n_obs) || n_obs == 0) {
    n_obs <- 1L
  }
  
  warning("augment() for autodiffr_fit objects is not yet fully implemented. ",
          "Fitted values and residuals are set to NA. ",
          "Model-specific augment methods may be added in future versions.")
  
  # If data is provided, attach it to the result
  if (!is.null(data)) {
    data_tbl <- tibble::as_tibble(data)
    # Combine data columns with fitted/residual columns
    # Create a list with all columns
    result_list <- as.list(data_tbl)
    result_list$.fitted <- rep(NA_real_, n_obs)
    result_list$.resid <- rep(NA_real_, n_obs)
    result <- do.call(tibble::tibble, result_list)
  } else {
    result <- tibble::tibble(
      .fitted = rep(NA_real_, n_obs),
      .resid = rep(NA_real_, n_obs)
    )
  }
  
  return(result)
}

