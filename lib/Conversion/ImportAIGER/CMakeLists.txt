add_circt_translation_library(CIRCTImportAIGER
  ImportAIGER.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTHW
  CIRCTComb
  CIRCTSeq
  CIRCTSupport
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
