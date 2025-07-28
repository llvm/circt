add_circt_unittest(CIRCTRTGTests
  MaterializerTest.cpp
  OpTests.cpp
)

target_link_libraries(CIRCTRTGTests
  PRIVATE
  CIRCTRTGDialect
  MLIRIR
)
