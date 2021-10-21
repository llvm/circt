get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(circt_dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)

add_llvm_executable(hlt-wrapgen
  BaseWrapper.cpp
  WrapGen.cpp
  StdWrapper.cpp
  HandshakeVerilatorWrapper.cpp
  CEmitterUtils.cpp
  VerilatorEmitterUtils.cpp
)

llvm_update_compile_flags(hlt-wrapgen)
target_link_libraries(hlt-wrapgen PRIVATE
  ${mlir_dialect_libs}
  ${circt_dialect_libs}
  )

