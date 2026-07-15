add_circt_unittest(CIRCTSystemCTests
  SystemCOpsTest.cpp
)

target_link_libraries(CIRCTSystemCTests
  PRIVATE
  CIRCTHW
  CIRCTSystemC
)
