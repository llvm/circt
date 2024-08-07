add_circt_dialect_library(CIRCTSimTransforms
  LowerDPIFunc.cpp
  ProceduralizeSim.cpp


  DEPENDS
  CIRCTSimTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSim
  CIRCTSeq
  CIRCTSV
  CIRCTComb
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRTransformUtils
)
