set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  ${mlir_dialect_libs}

  MLIRIR
  MLIRBytecodeReader
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
)

add_circt_tool(firld firld.cpp DEPENDS ${LIBS})
target_link_libraries(firld PRIVATE ${LIBS})

llvm_update_compile_flags(firld)
mlir_check_all_link_libraries(firld)
