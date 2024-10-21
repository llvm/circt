set(LLVM_LINK_COMPONENTS Support)

add_circt_tool(rtgtool rtgtool.cpp)
target_link_libraries(rtgtool
  PRIVATE
  CIRCTEmitRTGAssembly
  CIRCTRTGTestDialect
  CIRCTRTGDialect
  CIRCTRTGTransforms
  CIRCTSupport
  MLIRArithDialect
  MLIRIR
  LLVMSupport
)

llvm_update_compile_flags(rtgtool)
mlir_check_all_link_libraries(rtgtool)
