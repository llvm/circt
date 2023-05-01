add_circt_conversion_library(CIRCTSeqToSV
  LowerSeqToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSeq
  CIRCTSV
  MLIRTransforms
)
