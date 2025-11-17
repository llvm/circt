add_circt_conversion_library(CIRCTConvertToLLVM
  ConvertToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTCombToArith
  CIRCTCombToLLVM
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

