# Package onLoad hook
.onLoad <- function(libname, pkgname) {
  # Register autoplot method for autodiffr_fit if ggplot2 is loaded
  # We check if the generic exists (i.e., ggplot2 is attached)
  if (isNamespaceLoaded("ggplot2")) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        registerS3method("autoplot", "autodiffr_fit", autoplot.autodiffr_fit, 
                         envir = asNamespace(pkgname))
      }
    }, error = function(e) {
      # Silently fail if registration doesn't work
    })
  }
}

# Also register on attach if ggplot2 becomes available
.onAttach <- function(libname, pkgname) {
  if (isNamespaceLoaded("ggplot2")) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        registerS3method("autoplot", "autodiffr_fit", autoplot.autodiffr_fit, 
                         envir = asNamespace(pkgname))
      }
    }, error = function(e) {
      # Silently fail if registration doesn't work
    })
  }
}

