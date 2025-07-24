add_circt_unittest(CIRCTSupportTests
  FVIntTest.cpp
  JSONTest.cpp
  NPNClassTest.cpp
  PrettyPrinterTest.cpp
)

target_link_libraries(CIRCTSupportTests
  PRIVATE
  CIRCTSupport
)
