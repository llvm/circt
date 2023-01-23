set(LLVM_OPTIONAL_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  CPSATSchedulers.cpp
  LPSchedulers.cpp
  PresburgerSchedulers.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  TestPasses.cpp
  Utilities.cpp
  )

set(SCHEDULING_SOURCES
  ASAPScheduler.cpp
  ChainingSupport.cpp
  PresburgerSchedulers.cpp
  Problems.cpp
  SimplexSchedulers.cpp
  Utilities.cpp
  )

set(SCHEDULING_LIBS
  MLIRIR
  MLIRFuncDialect
  MLIRSupport
  )

if(ortools_FOUND)
  list(APPEND SCHEDULING_SOURCES LPSchedulers.cpp CPSATSchedulers.cpp)
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

if(ortools_FOUND)
  target_compile_definitions(obj.CIRCTSchedulingTestPasses PRIVATE SCHEDULING_OR_TOOLS)
endif()
