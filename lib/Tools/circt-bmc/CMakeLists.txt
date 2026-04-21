add_circt_library(CIRCTBMCTransforms
  ExternalizeRegisters.cpp
  LowerToBMC.cpp

  DEPENDS
  CIRCTBMCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  CIRCTComb
  CIRCTVerif
  CIRCTDebug

  MLIRFuncDialect
  MLIRIR
  MLIRLLVMDialect
  MLIRSupport
  MLIRSCFDialect
  MLIRTransforms
)
