add_circt_unittest(CIRCTSMTTests
  AttributeTest.cpp
  QuantifierTest.cpp
  TypeTest.cpp
)

target_link_libraries(CIRCTSMTTests
  PRIVATE
  CIRCTSMT
)
