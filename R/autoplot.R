#' Autoplot method for autodiffr_fit objects
#'
#' Creates diagnostic plots for fitted models. Right now just shows parameter
#' estimates with confidence intervals. More plot types will be added later
#' once we start storing optimization traces.
#'
#' @param object An object of class `autodiffr_fit`
#' @param type Character string specifying the type of plot. Currently only
#'   `"summary"` is supported, which shows a summary of the fit.
#' @param ... Additional arguments passed to plotting functions
#'
#' @return A ggplot object (if ggplot2 is available) or NULL
#'
#' @details
#' Requires ggplot2 (in Suggests). Eventually we'll add plots for optimization
#' traces (objective value, gradient norms, parameter paths) but that needs
#' us to store that info during fitting first.
#'
#' @export
#'
#' @examples
#' \dontrun{
#' library(ggplot2)
#' 
#' fit <- optim_mle(loglik, start, data)
#' autoplot(fit)
#' }
autoplot.autodiffr_fit <- function(object, type = c("summary"), ...) {
  type <- match.arg(type)
  
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    stop("ggplot2 package is required for autoplot() method. ",
         "Install it with: install.packages('ggplot2')")
  }
  
  if (type == "summary") {
    # Create a simple summary plot
    # For now, just show coefficients with confidence intervals if vcov is available
    
    vcov_mat <- stats::vcov(object)
    coefs <- stats::coef(object)
    
    if (length(coefs) == 0) {
      warning("No coefficients to plot.")
      return(NULL)
    }
    
    # Prepare data for plotting
    term <- names(coefs)
    if (is.null(term) || any(term == "")) {
      term <- paste0("par", seq_along(coefs))
    }
    
    plot_data <- data.frame(
      term = term,
      estimate = coefs,
      stringsAsFactors = FALSE
    )
    
    # Add confidence intervals if vcov is available
    if (!is.null(vcov_mat)) {
      se <- sqrt(diag(vcov_mat))
      plot_data$std.error <- se
      plot_data$conf.low <- coefs - 1.96 * se
      plot_data$conf.high <- coefs + 1.96 * se
    } else {
      plot_data$std.error <- NA_real_
      plot_data$conf.low <- NA_real_
      plot_data$conf.high <- NA_real_
    }
    
    # Create plot
    p <- ggplot2::ggplot(plot_data, ggplot2::aes_string(x = "term", y = "estimate")) +
      ggplot2::geom_point(size = 2) +
      ggplot2::geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
      ggplot2::labs(
        title = "Parameter Estimates",
        x = "Parameter",
        y = "Estimate"
      ) +
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
    
    # Add error bars if available
    if (!is.null(vcov_mat) && !any(is.na(plot_data$conf.low))) {
      p <- p + ggplot2::geom_errorbar(
        ggplot2::aes_string(ymin = "conf.low", ymax = "conf.high"),
        width = 0.2
      )
    }
    
    return(p)
  }
  
  return(NULL)
}

