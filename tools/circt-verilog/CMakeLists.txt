set(libs
  CIRCTComb
  CIRCTDebug
  CIRCTHW
  CIRCTImportVerilog
  CIRCTLLHD
  CIRCTLLHDTransforms
  CIRCTMoore
  CIRCTMooreToCore
  CIRCTMooreTransforms
  CIRCTSeq
  CIRCTSeqTransforms
  CIRCTSupport
  CIRCTTransforms
  CIRCTVerif
  MLIRBytecodeReader
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRFuncInlinerExtension
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSCFDialect
  MLIRSupport
)

add_circt_tool(circt-verilog circt-verilog.cpp DEPENDS ${libs})
target_link_libraries(circt-verilog PRIVATE ${libs})

llvm_update_compile_flags(circt-verilog)
mlir_check_all_link_libraries(circt-verilog)
