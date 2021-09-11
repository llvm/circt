set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  ComponentLibrary.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  TestPasses.cpp
  )

add_circt_library(CIRCTScheduling
  ASAPScheduler.cpp
  Problems.cpp
  SimplexSchedulers.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRStandard
  MLIRSupport
  )

add_circt_library(CIRCTComponentLibrary
  ComponentLibrary.cpp

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  )


add_circt_library(CIRCTSchedulingTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTScheduling
  CIRCTComponentLibrary
  MLIRPass
  )
