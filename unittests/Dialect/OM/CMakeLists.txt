add_subdirectory(Evaluator)

add_circt_unittest(CIRCTOMTests
  OMOpsTests.cpp
)

target_link_libraries(CIRCTOMTests
  PRIVATE
  CIRCTOM
  MLIRIR
)
