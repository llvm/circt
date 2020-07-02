add_mlir_conversion_library(MLIRLLHDToLLVM
  LLHDToLLVM.cpp

  DEPENDS
  MLIRLLHDConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRLLHD
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
)
