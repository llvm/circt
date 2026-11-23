add_circt_dialect_library(CIRCTLLHDTransforms
  CaptureUtils.cpp
  CombineDrives.cpp
  Deseq.cpp
  DeseqUtils.cpp
  HoistNets.cpp
  HoistSignals.cpp
  InlineCalls.cpp
  LowerProcesses.cpp
  Mem2Reg.cpp
  ProjectionUtils.cpp
  RemoveControlFlow.cpp
  Sig2Reg.cpp
  UndrivenToSymbolic.cpp
  UnrollLoops.cpp
  WrapProceduralOps.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTLLHD
  CIRCTSeq
  CIRCTVerif
  MLIRAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRSCFDialect
  MLIRTransformUtils
  MLIRUBDialect
)
