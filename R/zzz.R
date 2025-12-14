# Register autoplot method when package loads
.onLoad <- function(libname, pkgname) {
  # Try to register autoplot method if ggplot2 is available
  if (isNamespaceLoaded("ggplot2")) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        registerS3method("autoplot", "autodiffr_fit", autoplot.autodiffr_fit, 
                         envir = asNamespace(pkgname))
      }
    }, error = function(e) {
      # Registration failed, that's okay
    })
  }
}

.onAttach <- function(libname, pkgname) {
  # Same thing on attach, in case ggplot2 gets loaded later
  if (isNamespaceLoaded("ggplot2")) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        registerS3method("autoplot", "autodiffr_fit", autoplot.autodiffr_fit, 
                         envir = asNamespace(pkgname))
      }
    }, error = function(e) {
      # Registration failed, that's okay
    })
  }
}

