add_circt_conversion_library(CIRCTConvertToLLVM
  ConvertToLLVM.cpp
  HWToLLVM.cpp
  CombToLLVM.cpp
  LLHDToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTHW
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
)
