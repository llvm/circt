add_circt_unittest(CIRCTConversionUtilsTests
  SVLoweringUtilsTest.cpp
)

target_link_libraries(CIRCTConversionUtilsTests
  PRIVATE
  CIRCTEmit
  CIRCTHW
  CIRCTSV
  CIRCTSVLoweringUtils
)
