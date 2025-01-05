set(libs
  CIRCTVerilogLspServerLib
  MLIRBytecodeReader
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSCFDialect
  MLIRSupport
)

add_circt_tool(circt-verilog-lsp circt-verilog-lsp.cpp DEPENDS ${libs})
target_link_libraries(circt-verilog-lsp PRIVATE ${libs})

llvm_update_compile_flags(circt-verilog-lsp)
mlir_check_all_link_libraries(circt-verilog-lsp)
