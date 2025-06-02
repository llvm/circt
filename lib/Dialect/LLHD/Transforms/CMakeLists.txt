add_circt_dialect_library(CIRCTLLHDTransforms
  Deseq.cpp
  DesequentializationPass.cpp
  DeseqUtils.cpp
  EarlyCodeMotionPass.cpp
  FunctionEliminationPass.cpp
  HoistSignals.cpp
  LowerProcesses.cpp
  Mem2Reg.cpp
  MemoryToBlockArgumentPass.cpp
  ProcessLoweringPass.cpp
  Sig2RegPass.cpp
  TemporalCodeMotionPass.cpp
  TemporalRegions.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  CIRCTSeq
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRTransformUtils
)
