#' Constraint Specification for Parameters
#'
#' These functions create constraint specifications that can be used in
#' `optim_mle()` to enforce parameter constraints via smooth reparameterizations.
#'
#' @param name Character string naming the parameter(s) to constrain
#' @param k For `simplex()`, the number of elements in the simplex
#' @param dim For `corr_matrix()`, the dimension of the correlation matrix
#' @param ... For `constraints()`, constraint objects to combine
#'
#' @return An object of class `autodiffr_constraint` or `autodiffr_constraints`
#'
#' @details
#' Constraints are enforced via smooth reparameterizations:
#' \itemize{
#'   \item \code{positive()}: Uses exponential transformation, so unconstrained
#'     parameter `u` maps to constrained `theta = exp(u)`
#'   \item \code{simplex()}: Uses softmax transformation, so unconstrained
#'     parameters `u` map to constrained `theta = softmax(u)` (non-negative and sum to 1)
#'   \item \code{corr_matrix()}: Not yet implemented (will use Cholesky decomposition)
#' }
#'
#' When constraints are applied, the log-likelihood is adjusted by the log-Jacobian
#' of the transformation to maintain correct inference.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' # Positive constraint for variance parameter
#' constr <- positive("sigma")
#'
#' # Simplex constraint for mixture weights
#' constr <- simplex("weights", k = 3)
#'
#' # Combine constraints
#' constr_list <- constraints(
#'   positive("sigma"),
#'   simplex("weights", k = 3)
#' )
#' }
positive <- function(name) {
  structure(
    list(name = name, type = "positive"),
    class = "autodiffr_constraint"
  )
}

#' @rdname positive
#' @export
simplex <- function(name, k) {
  if (!is.numeric(k) || length(k) != 1 || k < 2) {
    stop("k must be a single integer >= 2")
  }
  structure(
    list(name = name, type = "simplex", k = as.integer(k)),
    class = "autodiffr_constraint"
  )
}

#' @rdname positive
#' @export
corr_matrix <- function(name, dim) {
  stop("corr_matrix constraints are not yet implemented. ",
       "This will use Cholesky decomposition in a future version.")
}

#' @rdname positive
#' @export
constraints <- function(...) {
  constraint_list <- list(...)
  # Validate all are constraint objects
  for (i in seq_along(constraint_list)) {
    if (!inherits(constraint_list[[i]], "autodiffr_constraint")) {
      stop("All arguments to constraints() must be constraint objects")
    }
  }
  structure(constraint_list, class = "autodiffr_constraints")
}

