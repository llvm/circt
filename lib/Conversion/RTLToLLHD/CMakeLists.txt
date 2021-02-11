add_circt_conversion_library(CIRCTRTLToLLHD
  RTLToLLHD.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTRTL
  MLIRTransforms
)
