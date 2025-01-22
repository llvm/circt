add_circt_tool(circt-synth circt-synth.cpp)
target_link_libraries(circt-synth
  PRIVATE
  CIRCTAIG
  CIRCTAIGToComb
  CIRCTAIGTransforms
  CIRCTAIGAnalysis
  CIRCTComb
  CIRCTCombToAIG
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTVerif
  CIRCTSV
  CIRCTEmit
  CIRCTSeq
  CIRCTOM
  CIRCTLTL
  CIRCTDebug
  CIRCTSim
  MLIRIR
  MLIRParser
  LLVMSupport
)

llvm_update_compile_flags(circt-synth)
mlir_check_all_link_libraries(circt-synth)
