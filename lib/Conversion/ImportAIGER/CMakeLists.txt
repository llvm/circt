add_circt_translation_library(CIRCTImportAIGER
  ImportAIGER.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTSupport
  CIRCTSynth
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
