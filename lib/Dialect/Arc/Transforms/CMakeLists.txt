add_circt_dialect_library(CIRCTArcTransforms
  Dedup.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
