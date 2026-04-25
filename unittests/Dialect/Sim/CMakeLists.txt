add_circt_unittest(CIRCTSimTests
  CascadeDeleteTest.cpp
)

target_link_libraries(CIRCTSimTests
  PRIVATE
  CIRCTHW
  CIRCTSim
  MLIRIR
  MLIRParser
)
