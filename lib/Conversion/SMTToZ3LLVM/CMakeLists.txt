add_circt_conversion_library(CIRCTSMTToZ3LLVM
  LowerSMTToZ3LLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSMT
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRTransforms
)
