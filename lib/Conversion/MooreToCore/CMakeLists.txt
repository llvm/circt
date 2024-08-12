add_circt_conversion_library(CIRCTMooreToCore
  MooreToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  CIRCTMoore
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRTransforms
)
