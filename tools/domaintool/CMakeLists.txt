set(libs
  CIRCTComb
  CIRCTDebug
  CIRCTEmit
  CIRCTHW
  CIRCTOM
  CIRCTOMEvaluator
  CIRCTSeq
  CIRCTSV
  CIRCTVerif
  MLIRParser
)

add_circt_tool(domaintool
  domaintool.cpp
  ClockSpecJSONHandler.cpp

  DEPENDS
  ${libs}
)
target_link_libraries(domaintool
  PRIVATE
  ${libs}
)

llvm_update_compile_flags(domaintool)
mlir_check_all_link_libraries(domaintool)
