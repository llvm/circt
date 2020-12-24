add_circt_conversion_library(CIRCTFIRRTLToRTL
  LowerToRTL.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  CIRCTRTL
  CIRCTSV
  MLIRTransforms
)
