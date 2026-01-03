add_circt_unittest(CIRCTArcRuntimeUnitTests
  ArcRuntimeTest.cpp
)

target_include_directories(CIRCTArcRuntimeUnitTests PRIVATE ${CIRCT_MAIN_INCLUDE_DIR}/circt/Tools/arcilator/)

target_link_libraries(CIRCTArcRuntimeUnitTests
  PRIVATE
  ArcJITRuntime
)
