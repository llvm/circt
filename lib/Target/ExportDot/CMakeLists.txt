add_circt_translation_library(CIRCTExportDot
  ExportDot.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTDC
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRTranslateLib
)
