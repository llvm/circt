set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS
  ${dialect_libs}
  ${mlir_dialect_libs}

  CIRCTArcReductions
  CIRCTEmitReductions
  CIRCTHWReductions
  CIRCTFIRRTLReductions
  CIRCTRTGReductions
  CIRCTReduceLib
  MLIRIR
  MLIRBytecodeWriter
  MLIRParser
  MLIRSupport
  MLIRTransforms
  MLIRReduceLib
)

add_circt_tool(circt-reduce
  circt-reduce.cpp
  DEPENDS ${LIBS}
)
target_link_libraries(circt-reduce PRIVATE ${LIBS})

llvm_update_compile_flags(circt-reduce)
mlir_check_all_link_libraries(circt-reduce)
