add_circt_dialect_library(CIRCTDatapathTransforms
  ReduceDelay.cpp

  DEPENDS
  CIRCTDatapathTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTDatapath
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)