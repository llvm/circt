add_circt_conversion_library(CIRCTLLHDToLLVM
  LLHDToLLVM.cpp

  DEPENDS
  CIRCTLLHDConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
)
