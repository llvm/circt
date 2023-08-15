add_llvm_executable(circt-capi-ir-test
  PARTIAL_SOURCES_INTENDED
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
  CIRCTCAPIExportFIRRTL
  CIRCTCAPIExportVerilog
)

add_llvm_executable(circt-capi-om-test
  PARTIAL_SOURCES_INTENDED
  om.c
)
llvm_update_compile_flags(circt-capi-om-test)

target_link_libraries(circt-capi-om-test
  PRIVATE

  MLIRCAPIIR
  CIRCTCAPIOM
)

add_llvm_executable(circt-capi-firrtl-test
  PARTIAL_SOURCES_INTENDED
  firrtl.c
)
llvm_update_compile_flags(circt-capi-firrtl-test)

target_link_libraries(circt-capi-firrtl-test
  PRIVATE

  MLIRCAPIIR
  CIRCTCAPIFIRRTL
  CIRCTCAPIExportFIRRTL
)
