add_circt_conversion_library(CIRCTHWArithToHW
  HWArithToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTSeq
  CIRCTHW
  CIRCTHWArith
  MLIRTransforms
)
