add_circt_conversion_library(CIRCTSMTToZ3LLVM
  LowerSMTToZ3LLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRLLVMCommonConversion
  MLIRSCFToControlFlow
  MLIRControlFlowToLLVM
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRSMT
  MLIRTransforms
)
