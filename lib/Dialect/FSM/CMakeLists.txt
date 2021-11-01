add_circt_dialect_library(CIRCTFSM
  FSMDialect.cpp
  FSMOps.cpp
  FSMTypes.cpp

  DEPENDS
  MLIRFSMIncGen

  LINK_LIBS PUBLIC
  MLIRStandard
  MLIRIR
  )
