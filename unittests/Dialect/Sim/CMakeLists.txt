add_circt_unittest(CIRCTSimTests
  TypesTest.cpp
)

target_link_libraries(CIRCTSimTests
  PRIVATE
  CIRCTSim
)
