
add_circt_translation_library(CIRCTExportVerilog
  ExportVerilog.cpp
  PrepareForEmission.cpp

  ADDITIONAL_HEADER_DIRS

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSV
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTranslation
  )
