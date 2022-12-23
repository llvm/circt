add_circt_dialect_library(CIRCTHWTransforms
  FlattenIO.cpp
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp
  InnerSymbolDCE.cpp
  PrintHWModuleGraph.cpp

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
