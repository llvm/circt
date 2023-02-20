add_circt_dialect_library(CIRCTArcTransforms
  Dedup.cpp
  InferMemories.cpp
  InlineModules.cpp
  MakeTables.cpp
  RemoveUnusedArcArguments.cpp
  SimplifyVariadicOps.cpp
  SinkInputs.cpp
  SplitLoops.cpp
  StripSV.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
