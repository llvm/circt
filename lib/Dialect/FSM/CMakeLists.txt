add_circt_dialect_library(CIRCTFSM
  FSMDialect.cpp
  FSMGraph.cpp
  FSMOps.cpp
  FSMTypes.cpp

  DEPENDS
  MLIRArithDialect
  MLIRFSMIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  CIRCTSeq
  MLIRFuncDialect
  MLIRArithDialect
  )

add_subdirectory(Transforms)
