add_circt_dialect_library(CIRCTHWTransforms
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp
  PrintHWModuleGraph.cpp
  FlattenIO.cpp
  MergeIdenticalPorts.cpp

  DEPENDS
  CIRCTHWTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTSeq
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
