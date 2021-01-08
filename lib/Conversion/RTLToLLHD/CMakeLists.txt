add_circt_conversion_library(CIRCTRTLToLLHD
  RTLToLLHD.cpp

  DEPENDS
  CIRCTRTLToLLHDConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTRTL
  MLIRTransforms
)
