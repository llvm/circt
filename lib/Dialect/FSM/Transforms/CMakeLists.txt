add_circt_dialect_library(CIRCTFSMTransforms
  PrintFSMGraph.cpp
  RemoveUnreachableStates.cpp

  DEPENDS
  CIRCTFSMTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFSM
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
