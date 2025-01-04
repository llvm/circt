add_circt_tool(circt-vcd circt-vcd.cpp)

target_link_libraries(circt-vcd
  PRIVATE
  CIRCTComb
  CIRCTHW
  CIRCTVerif
  CIRCTLTL
  CIRCTSV
  CIRCTOM
  CIRCTEmit
  CIRCTSupport
  MLIRIR
  MLIRParser
  LLVMSupport
)

llvm_update_compile_flags(circt-vcd)
mlir_check_all_link_libraries(circt-vcd)
