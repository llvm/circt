add_circt_dialect_library(CIRCTLLHDTransforms
  CombineDrives.cpp
  Deseq.cpp
  DesequentializationPass.cpp
  DeseqUtils.cpp
  EarlyCodeMotionPass.cpp
  HoistSignals.cpp
  InlineCalls.cpp
  LowerProcesses.cpp
  Mem2Reg.cpp
  MemoryToBlockArgumentPass.cpp
  ProcessLoweringPass.cpp
  RemoveControlFlow.cpp
  SelectToMux.cpp
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
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRTransformUtils
)
