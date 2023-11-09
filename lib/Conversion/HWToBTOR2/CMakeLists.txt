add_circt_conversion_library(CIRCTHWToBTOR2
  HWToBTOR2.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTComb
  CIRCTSeq
  MLIRTransforms
)
