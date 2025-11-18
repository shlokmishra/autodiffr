#' Create an autodiffr fit object
#'
#' @param coefficients Named numeric vector of parameter estimates
#' @param loglik Final log-likelihood value
#' @param convergence Convergence code (0 = success)
#' @param message Convergence message
#' @param iterations Number of iterations
#' @param gradient_norm Norm of the final gradient
#' @param gradient Named numeric vector of final gradients
#' @param optimizer Character string naming the optimizer used
#' @param call The original function call
#'
#' @return An object of class `autodiffr_fit`
#'
#' @keywords internal
new_autodiffr_fit <- function(coefficients = numeric(0),
                               loglik = NA_real_,
                               convergence = NA_integer_,
                               message = character(0),
                               iterations = NA_integer_,
                               gradient_norm = NA_real_,
                               gradient = numeric(0),
                               optimizer = character(0),
                               call = NULL) {
  structure(
    list(
      coefficients = coefficients,
      loglik = loglik,
      convergence = convergence,
      message = message,
      iterations = iterations,
      gradient_norm = gradient_norm,
      gradient = gradient,
      optimizer = optimizer,
      call = call
    ),
    class = "autodiffr_fit"
  )
}

#' Constructor for autodiffr fit objects
#'
#' @param coefficients Named numeric vector of parameter estimates
#' @param loglik Final log-likelihood value
#' @param convergence Convergence code (0 = success)
#' @param message Convergence message
#' @param iterations Number of iterations
#' @param gradient_norm Norm of the final gradient
#' @param gradient Named numeric vector of final gradients
#' @param optimizer Character string naming the optimizer used
#' @param call The original function call
#'
#' @return An object of class `autodiffr_fit`
#'
#' @keywords internal
autodiffr_fit <- function(coefficients,
                          loglik,
                          convergence,
                          message,
                          iterations,
                          gradient_norm,
                          gradient,
                          optimizer,
                          call) {
  new_autodiffr_fit(
    coefficients = coefficients,
    loglik = loglik,
    convergence = convergence,
    message = message,
    iterations = iterations,
    gradient_norm = gradient_norm,
    gradient = gradient,
    optimizer = optimizer,
    call = call
  )
}

