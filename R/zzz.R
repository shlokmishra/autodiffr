# Helper function to register broom methods
# This is internal but can be called if needed
register_broom_methods <- function(pkgname = "autodiffr") {
  if (requireNamespace("generics", quietly = TRUE)) {
    tryCatch({
      registerS3method("tidy", "autodiffr_fit", tidy.autodiffr_fit, 
                       envir = asNamespace(pkgname))
      registerS3method("glance", "autodiffr_fit", glance.autodiffr_fit, 
                       envir = asNamespace(pkgname))
      registerS3method("augment", "autodiffr_fit", augment.autodiffr_fit, 
                       envir = asNamespace(pkgname))
    }, error = function(e) {
      # Registration failed, that's okay
    })
  }
}

# Register S3 methods for optional dependencies when package loads
.onLoad <- function(libname, pkgname) {
  # Register autoplot method if ggplot2 is available
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
  
  # Register broom methods if generics is available
  register_broom_methods(pkgname)
  
  # Set hook to register when broom/generics is attached
  setHook(packageEvent("generics", "onLoad"), function(...) {
    register_broom_methods(pkgname)
  })
  setHook(packageEvent("broom", "onLoad"), function(...) {
    register_broom_methods(pkgname)
  })
}

.onAttach <- function(libname, pkgname) {
  # Register autoplot if ggplot2 gets loaded later
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
  
  # Register broom methods if they're available
  register_broom_methods(pkgname)
}

