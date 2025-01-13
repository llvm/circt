add_circt_unittest(CIRCTRTGTestTests
  RegisterTest.cpp
)

target_link_libraries(CIRCTRTGTestTests
  PRIVATE
  CIRCTRTGTestDialect
)
