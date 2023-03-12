add_circt_dialect_library(CIRCTArcTransforms
  Dedup.cpp
  InlineModules.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
