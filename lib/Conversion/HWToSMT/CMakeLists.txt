add_circt_conversion_library(CIRCTHWToSMT
  HWToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSMT
  CIRCTSeq
  MLIRFuncDialect
  MLIRTransforms
  MLIRTransformUtils
)
