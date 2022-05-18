add_circt_dialect_library(CIRCTSeqTransforms
  LowerSeqToSV.cpp

  DEPENDS
  CIRCTSeqTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
