add_circt_dialect_library(CIRCTHWTransforms
  HWPrintInstanceGraph.cpp

  DEPENDS
  CIRCTHWTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
