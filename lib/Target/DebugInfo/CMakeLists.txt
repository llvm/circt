add_circt_translation_library(CIRCTTargetDebugInfo
  DumpDebugInfo.cpp
  EmitHGLDD.cpp
  TranslateRegistration.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebugInfoAnalysis
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRTranslateLib
)
