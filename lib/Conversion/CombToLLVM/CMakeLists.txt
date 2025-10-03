add_circt_conversion_library(CIRCTCombToLLVM
  CombToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTCombToArith
  CIRCTHW
  CIRCTHWToLLVM
  CIRCTSupport
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRTransforms
)
