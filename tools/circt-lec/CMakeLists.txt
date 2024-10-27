if(MLIR_ENABLE_EXECUTION_ENGINE)
  add_compile_definitions(CIRCT_LEC_ENABLE_JIT)
  set(CIRCT_LEC_JIT_LLVM_COMPONENTS native)
  set(CIRCT_LEC_JIT_DEPS
    MLIRExecutionEngine
    MLIRExecutionEngineUtils
  )
endif()

set(LLVM_LINK_COMPONENTS Support ${CIRCT_LEC_JIT_LLVM_COMPONENTS})

add_circt_tool(circt-lec circt-lec.cpp)
target_link_libraries(circt-lec
  PRIVATE
  CIRCTLECTransforms
  CIRCTSMTToZ3LLVM
  CIRCTHWToSMT
  CIRCTCombToSMT
  CIRCTVerifToSMT
  CIRCTComb
  CIRCTHW
  CIRCTSMT
  CIRCTSupport
  CIRCTVerif
  MLIRIR
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLLVMIRTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRTargetLLVMIRExport
  MLIRFuncInlinerExtension
  MLIRBuiltinToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  LLVMSupport

  ${CIRCT_LEC_JIT_DEPS}
)

llvm_update_compile_flags(circt-lec)
mlir_check_all_link_libraries(circt-lec)
