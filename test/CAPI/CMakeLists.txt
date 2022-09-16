add_llvm_executable(circt-capi-ir-test
  ir.c
)
llvm_update_compile_flags(circt-capi-ir-test)

target_link_libraries(circt-capi-ir-test
  PRIVATE

  CIRCTCAPIComb
  CIRCTCAPIHW
  CIRCTCAPISeq
  CIRCTCAPISV
  CIRCTCAPIFSM
  CIRCTCAPIExportVerilog
)
