add_circt_unittest(CIRCTRTGTestTests
  MaterializerTest.cpp
  RegisterAllocationInterfaceTest.cpp
)

target_link_libraries(CIRCTRTGTestTests
  PRIVATE
  CIRCTRTGDialect
  CIRCTRTGTestDialect
  MLIRIR
)
