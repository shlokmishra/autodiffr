new_autodiffr_fit <- function(coefficients = numeric(0),
                               loglik = NA_real_,
                               convergence = NA_integer_,
                               message = character(0),
                               iterations = NA_integer_,
                               gradient_norm = NA_real_,
                               gradient = numeric(0),
                               optimizer = character(0),
                               vcov = NULL,
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
      vcov = vcov,
      call = call
    ),
    class = "autodiffr_fit"
  )
}

autodiffr_fit <- function(coefficients,
                          loglik,
                          convergence,
                          message,
                          iterations,
                          gradient_norm,
                          gradient,
                          optimizer,
                          vcov = NULL,
                          call = NULL) {
  new_autodiffr_fit(
    coefficients = coefficients,
    loglik = loglik,
    convergence = convergence,
    message = message,
    iterations = iterations,
    gradient_norm = gradient_norm,
    gradient = gradient,
    optimizer = optimizer,
    vcov = vcov,
    call = call
  )
}


