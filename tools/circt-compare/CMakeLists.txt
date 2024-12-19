add_circt_tool(circt-compare circt-compare.cpp)
target_link_libraries(circt-compare
  PRIVATE
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTLTL
  CIRCTSim
  CIRCTEmit
  CIRCTOM
  CIRCTVerif
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRFuncInlinerExtension
  MLIRBuiltinToLLVMIRTranslation
  LLVMSupport

  ${CIRCT_BMC_JIT_DEPS}
)

llvm_update_compile_flags(circt-compare)
mlir_check_all_link_libraries(circt-compare)
