add_circt_unittest(CIRCTSVTests
  SVOpsTest.cpp
)

target_link_libraries(CIRCTSVTests
  PRIVATE
  CIRCTHW
  CIRCTSV
)
