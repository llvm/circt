add_circt_translation_library(CIRCTExportSystemC
  ExportSystemC.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSystemC
  CIRCTSupport
  MLIRIR
  MLIRTranslateLib
)
