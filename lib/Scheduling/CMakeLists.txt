set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  LPSchedulers.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  TestPasses.cpp
  Utilities.cpp
  )

set(SCHEDULING_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_LIBS
  MLIRIR
  MLIRStandard
  MLIRSupport
  )

if(ortools_FOUND)
  option(SCHEDULING_OR_TOOLS "Enable schedulers relying on an external solver from OR-Tools" ON)
endif()

if(SCHEDULING_OR_TOOLS)
  list(APPEND SCHEDULING_SOURCES LPSchedulers.cpp)
  list(APPEND SCHEDULING_LIBS ortools::ortools)
endif()

add_circt_library(CIRCTScheduling
  ${SCHEDULING_SOURCES}

  LINK_LIBS PUBLIC
  ${SCHEDULING_LIBS}
  )

add_circt_library(CIRCTSchedulingTestPasses
  TestPasses.cpp

  LINK_LIBS PUBLIC
  CIRCTScheduling
  MLIRPass
  )

if(SCHEDULING_OR_TOOLS)
  target_compile_definitions(obj.CIRCTSchedulingTestPasses PRIVATE SCHEDULING_OR_TOOLS)
endif()
