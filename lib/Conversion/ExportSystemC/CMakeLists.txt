
add_circt_translation_library(CIRCTExportSystemC
  ExportSystemC.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSystemC
  CIRCTSupport
  MLIRPass
  MLIRSideEffectInterfaces
  )
