add_circt_translation_library(CIRCTExportAIGER
  ExportAIGER.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTSeq
  CIRCTSynth
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
