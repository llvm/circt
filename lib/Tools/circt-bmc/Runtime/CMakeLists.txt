add_circt_library(CIRCTBMCRuntime
  EXCLUDE_FROM_LIBMLIR
  BMCTrace.cpp

  LINK_COMPONENTS
  Support
)
