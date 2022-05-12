add_circt_dialect_library(CIRCTFSM
  FSMDialect.cpp
  FSMGraph.cpp
  FSMOps.cpp
  FSMTypes.cpp

  DEPENDS
  MLIRFSMIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFunc
  MLIRArithmetic
  )

add_subdirectory(Transforms)
