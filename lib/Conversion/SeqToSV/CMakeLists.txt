add_circt_conversion_library(CIRCTSeqToSV
  SeqToSV.cpp

  DEPENDS
  CIRCTHW
  CIRCTSV
  MLIRSeqIncGen
  MLIRSeqMiscIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTSV
  CIRCTSeq
  MLIRPass
  MLIRIR
  MLIRTransforms
)
