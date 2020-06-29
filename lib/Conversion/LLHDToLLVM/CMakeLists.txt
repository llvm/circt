add_mlir_conversion_library(MLIRLLHDToLLVM 
  LLHDToLLVM.cpp

  DEPENDS
  MLIRLLHDIncGen
  MLIRLLHDConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMIR
  MLIRStandardToLLVM
  MLIRVector
  MLIRTransforms
  LLVMCore
  LLVMSupport
)
