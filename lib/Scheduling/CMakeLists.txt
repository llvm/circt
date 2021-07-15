set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  Problems.cpp
  TestPasses.cpp
  )

add_circt_library(CIRCTScheduling
  ASAPScheduler.cpp
  Problems.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSupport
  )

add_circt_library(CIRCTSchedulingTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTScheduling
  MLIRPass
  )
