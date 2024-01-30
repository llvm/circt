add_circt_tool(circt-verilog
  circt-verilog.cpp
)

llvm_update_compile_flags(circt-verilog)

target_link_libraries(circt-verilog PRIVATE
  CIRCTImportVerilog
  CIRCTSupport
  MLIRIR
)
