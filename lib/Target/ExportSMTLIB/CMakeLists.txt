add_circt_translation_library(CIRCTExportSMTLIB
  ExportSMTLIB.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRSMT
  MLIRTranslateLib
)
