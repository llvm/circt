add_circt_dialect_library(CIRCTArcTransforms
  AddTaps.cpp
  Dedup.cpp
  InferMemories.cpp
  InlineArcs.cpp
  InlineModules.cpp
  LowerLUT.cpp
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
  CIRCTSV
  CIRCTSeq
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
