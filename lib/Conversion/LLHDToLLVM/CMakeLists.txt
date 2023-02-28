add_circt_conversion_library(CIRCTLLHDToLLVM
  LLHDToLLVM.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTCombToLLVM
  CIRCTHWToLLVM
  CIRCTHW
  CIRCTSupport
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
  )
