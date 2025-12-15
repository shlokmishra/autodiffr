# Helper function to register broom methods
# This is internal but can be called if needed
register_broom_methods <- function(pkgname = "autodiffr") {
  if (requireNamespace("generics", quietly = TRUE)) {
    tryCatch({
      # Get methods from namespace - they must exist
      # Use inherits = TRUE to allow finding them even if not fully loaded
      if (exists("tidy.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)) {
        tidy_method <- get("tidy.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)
        registerS3method("tidy", "autodiffr_fit", tidy_method, 
                         envir = asNamespace(pkgname))
      }
      if (exists("glance.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)) {
        glance_method <- get("glance.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)
        registerS3method("glance", "autodiffr_fit", glance_method, 
                         envir = asNamespace(pkgname))
      }
      if (exists("augment.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)) {
        augment_method <- get("augment.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)
        registerS3method("augment", "autodiffr_fit", augment_method, 
                         envir = asNamespace(pkgname))
      }
    }, error = function(e) {
      # Registration failed, that's okay - methods may not be available yet
      # This is non-fatal
    })
  }
}

# Register S3 methods for optional dependencies when package loads
.onLoad <- function(libname, pkgname) {
  # Don't try to register methods during .onLoad - namespace may not be fully loaded
  # Set hooks to register when packages are attached
  setHook(packageEvent("generics", "onLoad"), function(...) {
    register_broom_methods(pkgname)
  })
  setHook(packageEvent("broom", "onLoad"), function(...) {
    register_broom_methods(pkgname)
  })
  setHook(packageEvent("ggplot2", "onLoad"), function(...) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        if (exists("autoplot.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)) {
          autoplot_method <- get("autoplot.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)
          registerS3method("autoplot", "autodiffr_fit", autoplot_method, 
                           envir = asNamespace(pkgname))
        }
      }
    }, error = function(e) {
      # Registration failed, that's okay
    })
  })
}

.onAttach <- function(libname, pkgname) {
  # Register autoplot if ggplot2 gets loaded later
  if (isNamespaceLoaded("ggplot2")) {
    tryCatch({
      if (exists("autoplot", envir = asNamespace("ggplot2"))) {
        if (exists("autoplot.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)) {
          autoplot_method <- get("autoplot.autodiffr_fit", envir = asNamespace(pkgname), inherits = TRUE)
          registerS3method("autoplot", "autodiffr_fit", autoplot_method, 
                           envir = asNamespace(pkgname))
        }
      }
    }, error = function(e) {
      # Registration failed, that's okay
    })
  }
  
  # Register broom methods if they're available (now namespace is fully loaded)
  register_broom_methods(pkgname)
}

