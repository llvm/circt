add_circt_dialect_library(CIRCTIbisTransforms
  IbisCallPrep.cpp

  DEPENDS
  CIRCTIbisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDC
  CIRCTIbis
  CIRCTHW
  CIRCTSV
  CIRCTSeq
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
