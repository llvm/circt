add_circt_translation_library(CIRCTExportAIGER
  ExportAIGER.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSeq
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
