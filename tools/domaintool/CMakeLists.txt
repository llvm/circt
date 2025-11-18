get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_circt_tool(domaintool
  domaintool.cpp
  ClockSpecJSONHandler.cpp
)

target_link_libraries(domaintool
  PRIVATE
  ${dialect_libs}
  ${mlir_dialect_libs}
  CIRCTOMEvaluator
  MLIRParser
  MLIRSupport
)

llvm_update_compile_flags(domaintool)
mlir_check_all_link_libraries(domaintool)
