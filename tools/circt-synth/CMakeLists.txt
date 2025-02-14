add_circt_tool(circt-synth circt-synth.cpp)
target_link_libraries(circt-synth
  PRIVATE
  CIRCTAIG
  CIRCTAIGToComb
  CIRCTAIGTransforms
  CIRCTComb
  CIRCTCombToAIG
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSupport
  MLIRIR
  MLIRParser
  LLVMSupport
)

llvm_update_compile_flags(circt-synth)
mlir_check_all_link_libraries(circt-synth)
