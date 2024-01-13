add_circt_unittest(CIRCTSMTTests
  AttributeTest.cpp
)

target_link_libraries(CIRCTSMTTests
  PRIVATE
  CIRCTSMT
)
