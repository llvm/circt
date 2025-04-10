if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_compile_definitions(CIRCT_BMC_ENABLE_JIT)
  set(CIRCT_BMC_JIT_LLVM_COMPONENTS native)
  set(CIRCT_BMC_JIT_DEPS
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
  )
endif()

set(LLVM_LINK_COMPONENTS Support ${CIRCT_BMC_JIT_LLVM_COMPONENTS})

add_circt_tool(circt-bmc circt-bmc.cpp)
target_link_libraries(circt-bmc
  PRIVATE
  CIRCTBMCTransforms
  CIRCTComb
  CIRCTCombToSMT
  CIRCTEmitTransforms
  CIRCTHW
  CIRCTHWToSMT
  CIRCTOMTransforms
  CIRCTSeq
  CIRCTSMTToZ3LLVM
  CIRCTSupport
  CIRCTVerif
  CIRCTVerifToSMT
  LLVMSupport
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRSMT
  MLIRTargetLLVMIRExport

  ${CIRCT_BMC_JIT_DEPS}
)

llvm_update_compile_flags(circt-bmc)
mlir_check_all_link_libraries(circt-bmc)
