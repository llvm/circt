add_circt_unittest(CIRCTRTGTestTests
  MaterializerTest.cpp
)

target_link_libraries(CIRCTRTGTestTests
  PRIVATE
  CIRCTRTGTestDialect
  MLIRIR
)
