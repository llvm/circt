add_circt_conversion_library(CIRCTLLHDToLLVM
  LLHDToLLVM.cpp
  ../HWToLLVM/HWToLLVM.cpp
  ../CombToLLVM/CombToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTHW
  MLIRArithmeticToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
)
