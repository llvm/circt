set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  Interfaces.cpp
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

add_circt_library(CIRCTSchedulingInterfaces
  Interfaces.cpp

  DEPENDS
  MLIRSchedulingInterfacesIncGen
  )

add_circt_library(CIRCTSchedulingTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTScheduling
  CIRCTSchedulingInterfaces
  MLIRPass
  )
