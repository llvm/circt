add_circt_unittest(CIRCTCombTests
  KnownBitsTest.cpp
)

target_link_libraries(CIRCTCombTests
  PRIVATE
  CIRCTHW
  CIRCTComb
  MLIRIR
  MLIRParser
)
