add_circt_library(CIRCTSchedulingTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTScheduling
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  )
