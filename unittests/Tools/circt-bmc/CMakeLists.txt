add_circt_unittest(CIRCTBMCToolsTests
  BMCTraceTest.cpp
)

target_link_libraries(CIRCTBMCToolsTests
  PRIVATE
  CIRCTBMCRuntime
)
