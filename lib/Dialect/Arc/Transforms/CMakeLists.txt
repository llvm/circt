add_circt_dialect_library(CIRCTArcTransforms
  Dedup.cpp
  InlineModules.cpp
  SplitLoops.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
