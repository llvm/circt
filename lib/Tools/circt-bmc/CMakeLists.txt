add_circt_library(CIRCTBMCTransforms
  ExternalizeRegisters.cpp

  DEPENDS
  CIRCTBMCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq

  MLIRIR
  MLIRSupport
  MLIRSCFDialect
  MLIRTransforms
)
