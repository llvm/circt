add_circt_dialect_library(CIRCTSVTransforms
  RTLCleanup.cpp

  DEPENDS
  CIRCTSVTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
