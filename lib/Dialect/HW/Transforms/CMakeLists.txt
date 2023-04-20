add_circt_dialect_library(CIRCTHWTransforms
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp
  NarrowPorts.cpp
  PrintHWModuleGraph.cpp
  FlattenIO.cpp

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
