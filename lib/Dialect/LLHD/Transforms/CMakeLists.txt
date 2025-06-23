add_circt_dialect_library(CIRCTLLHDTransforms
  CombineDrives.cpp
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
  WrapProceduralOps.cpp

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
