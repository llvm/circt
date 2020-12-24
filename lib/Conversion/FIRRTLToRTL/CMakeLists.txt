add_circt_conversion_library(CIRCTFIRRTLToRTL
  LowerToRTL.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTRTL
  CIRCTSV
  MLIRTransforms
)
