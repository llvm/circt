set(libs
  CIRCTVerilogLspServerLib
)

add_circt_tool(circt-verilog-lsp-server circt-verilog-lsp-server.cpp DEPENDS ${libs})
target_link_libraries(circt-verilog-lsp-server PRIVATE ${libs})

llvm_update_compile_flags(circt-verilog-lsp-server)
mlir_check_all_link_libraries(circt-verilog-lsp-server)
