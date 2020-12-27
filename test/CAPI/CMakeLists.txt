set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_llvm_executable(circt-capi-ir-test
  ir.c
  )
llvm_update_compile_flags(circt-capi-ir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(circt-capi-ir-test
  PRIVATE
  CIRCTCAPIRTL
  ${dialect_libs}

  MLIRCAPIIR
  MLIRCAPIRegistration
  MLIRPublicAPI
  )
