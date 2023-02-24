add_circt_dialect_library(CIRCTArcTransforms
  Dedup.cpp
  InlineModules.cpp
  MakeTables.cpp
  SinkInputs.cpp
  SplitLoops.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTHW
  CIRCTComb
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
