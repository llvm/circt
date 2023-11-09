set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(fsm-verification fsm-verification.cpp)
target_link_libraries(fsm-verification
  PRIVATE
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
  CIRCTFSM
  CIRCTHW
  CIRCTOM
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncInlinerExtension
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRParser
  MLIRTargetLLVMIRExport
)

llvm_update_compile_flags(fsm-verification)
mlir_check_all_link_libraries(fsm-verification)