add_circt_conversion_library(CIRCTFIRRTLToLLHD
  FIRRTLToLLHD.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTFIRRTL
  MLIRTransforms
)
