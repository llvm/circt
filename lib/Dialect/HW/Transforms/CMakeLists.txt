add_circt_dialect_library(CIRCTHWTransforms
  HWFlattenStructs.cpp
  HWPrintInstanceGraph.cpp
  HWSpecialize.cpp

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
