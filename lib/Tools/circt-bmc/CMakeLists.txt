add_circt_library(CIRCTBMCTransforms
  ExternalizeRegisters.cpp
  LowerToBMC.cpp
  MaterializeDebugVariables.cpp

  DEPENDS
  CIRCTBMCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDebug
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTVerif

  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRSCFDialect
  MLIRTransforms
)
