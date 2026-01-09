add_circt_unittest(CIRCTArcRuntimeUnitTests
  ArcRuntimeTest.cpp
)

target_link_libraries(CIRCTArcRuntimeUnitTests
  PRIVATE
  CIRCTArcJITRuntime
)
