set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(fsm-verification fsm-verification.cpp)
target_link_libraries(fsm-verification
  PRIVATE
  CIRCTFSM
  CIRCTArc
  CIRCTArcToLLVM
  CIRCTArcTransforms
  CIRCTCombToArith
  CIRCTConvertToArcs
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
  ${Z3_LIBRARIES}

)

llvm_update_compile_flags(fsm-verification)
mlir_check_all_link_libraries(fsm-verification)