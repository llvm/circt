add_circt_unittest(CIRCTFIRRTLTests
  AttributesTest.cpp
  TypesTest.cpp
)

target_link_libraries(CIRCTFIRRTLTests
  PRIVATE
  CIRCTFIRRTL
)
