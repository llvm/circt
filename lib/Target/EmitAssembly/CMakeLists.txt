add_circt_translation_library(CIRCTEmitAssembly
  EmitAssembly.cpp

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
