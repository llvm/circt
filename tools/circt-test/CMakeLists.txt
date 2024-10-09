set(libs
  CIRCTComb
  CIRCTHW
  CIRCTOM
  CIRCTSeq
  CIRCTSim
  CIRCTSV
  CIRCTVerif

  MLIRLLVMDialect
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSCFDialect

  MLIRBytecodeReader
  MLIRIR
  MLIRParser
  MLIRSupport
)

add_circt_tool(circt-test circt-test.cpp DEPENDS ${libs})
target_link_libraries(circt-test PRIVATE ${libs})

llvm_update_compile_flags(circt-test)
mlir_check_all_link_libraries(circt-test)
