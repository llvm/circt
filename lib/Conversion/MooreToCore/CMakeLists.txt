add_circt_conversion_library(CIRCTMooreToCore
  MooreToCore.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTLLHD
  CIRCTMoore
  CIRCTSim
  CIRCTVerif
  CIRCTTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSideEffectInterfaces
  MLIRTransforms
)
