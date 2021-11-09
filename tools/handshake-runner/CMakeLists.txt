get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(handshake-runner handshake-runner.cpp Simulation.cpp)

llvm_update_compile_flags(handshake-runner)
target_link_libraries(handshake-runner PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  CIRCTHandshake
  CIRCTStandardToHandshake
  CIRCTHW
  CIRCTESI
  )
