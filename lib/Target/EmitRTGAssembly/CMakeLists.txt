add_circt_translation_library(CIRCTEmitRTGAssembly
  EmitRTGAssembly.cpp

  ADDITIONAL_HEADER_DIRS

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTRTGDialect
  CIRCTRTGTestDialect
  CIRCTSupport
  MLIRIR
  MLIRTranslateLib
)
