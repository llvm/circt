add_circt_unittest(CIRCTSupportTests
  PrettyPrinterTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
)
