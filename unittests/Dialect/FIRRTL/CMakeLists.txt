add_circt_unittest(CIRCTFIRRTLTests
  AttributesTest.cpp
  PortsTest.cpp
  TypesTest.cpp
)

target_link_libraries(CIRCTFIRRTLTests
  PRIVATE
  CIRCTFIRRTL
)
