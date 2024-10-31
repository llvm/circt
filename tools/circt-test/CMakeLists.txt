set(libs
  CIRCTComb
  CIRCTExportVerilog
  CIRCTHW
  CIRCTOM
  CIRCTSeq
  CIRCTSim
  CIRCTSV
  CIRCTSVTransforms
  CIRCTVerif
  CIRCTVerifToSV
  CIRCTVerifTransforms

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

configure_file(circt-test-runner-sby.py ${CIRCT_TOOLS_DIR}/circt-test-runner-sby.py)
add_custom_target(circt-test-runner-sby SOURCES ${CIRCT_TOOLS_DIR}/circt-test-runner-sby.py)
