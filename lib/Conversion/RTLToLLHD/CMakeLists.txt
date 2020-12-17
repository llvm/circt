add_circt_conversion_library(MLIRRTLToLLHD
  RTLToLLHD.cpp

  DEPENDS
  MLIRRTLToLLHDConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLHD
  MLIRRTL
  MLIRTransforms
)
