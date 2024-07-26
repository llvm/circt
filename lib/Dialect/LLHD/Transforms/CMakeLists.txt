add_circt_dialect_library(CIRCTLLHDTransforms
  DesequentializationPass.cpp
  EarlyCodeMotionPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  ProcessLoweringPass.cpp
  TemporalCodeMotionPass.cpp
  TemporalRegions.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTLLHD
  MLIRIR
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRTransformUtils
)
