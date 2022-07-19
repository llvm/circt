add_circt_conversion_library(CIRCTCOMBToLLVM
  CombToLLVM.cpp


  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTLLHD
  CIRCTComb
  CIRCTHW
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLLVMCommonConversion
  MLIRVectorDialect
  MLIRTransforms
  MLIRReconcileUnrealizedCasts
)
