add_llvm_executable(circt-capi-ir-test
  ir.c
  )
llvm_update_compile_flags(circt-capi-ir-test)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(circt-capi-ir-test
  PRIVATE
  ${dialect_libs}

#  MLIRCAPIRegistration
  MLIRPublicAPI
  CIRCTCAPIComb
  CIRCTCAPIRTL
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPIExportVerilog
  )
