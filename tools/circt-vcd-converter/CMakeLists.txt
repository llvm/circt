add_circt_tool(circt-vcd-converter circt-vcd-converter.cpp)
target_link_libraries(circt-vcd-converter
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

llvm_update_compile_flags(circt-vcd-converter)
mlir_check_all_link_libraries(circt-vcd-converter)
