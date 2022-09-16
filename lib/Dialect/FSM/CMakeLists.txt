add_circt_dialect_library(CIRCTFSM
  FSMDialect.cpp
  FSMGraph.cpp
  FSMOps.cpp
  FSMTypes.cpp

  DEPENDS
  MLIRFSMIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  MLIRFuncDialect
  MLIRArithmeticDialect
  )

add_subdirectory(Transforms)
