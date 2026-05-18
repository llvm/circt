add_circt_unittest(CIRCTRTGTests
  AttrTests.cpp
  OpTests.cpp
)

target_link_libraries(CIRCTRTGTests
  PRIVATE
  CIRCTRTGDialect
  CIRCTRTGTestDialect
  MLIRIR
)
