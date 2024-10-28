add_circt_tool(circt-synth circt-synth.cpp)
target_link_libraries(circt-synth
  PRIVATE
  CIRCTAIG
  CIRCTAIGAnalysis
  CIRCTAIGToComb
  CIRCTAIGTransforms
  CIRCTComb
  CIRCTCombToAIG
  CIRCTHW
  CIRCTOM
  CIRCTVerif
  CIRCTSeq
  CIRCTDebug
  CIRCTSim
  CIRCTSV
  CIRCTEmit
  CIRCTSupport
  CIRCTVerif
  MLIRIR
  MLIRParser
  LLVMSupport
)

llvm_update_compile_flags(circt-synth)
mlir_check_all_link_libraries(circt-synth)
