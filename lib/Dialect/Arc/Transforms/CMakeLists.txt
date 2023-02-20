add_circt_dialect_library(CIRCTArcTransforms
  AddTaps.cpp
  AllocateState.cpp
  Dedup.cpp
  InferMemories.cpp
  InferStateProperties.cpp
  InlineArcs.cpp
  InlineModules.cpp
  LegalizeStateUpdate.cpp
  LowerClocksToFuncs.cpp
  LowerLUT.cpp
  LowerState.cpp
  MakeTables.cpp
  MuxToControlFlow.cpp
  PrintStateInfo.cpp
  RemoveUnusedArcArguments.cpp
  SimplifyVariadicOps.cpp
  SinkInputs.cpp
  SplitLoops.cpp
  StripSV.cpp

  DEPENDS
  CIRCTArcTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTArc
  CIRCTArcExternalInterfaces
  CIRCTComb
  CIRCTHW
  CIRCTSV
  CIRCTSeq
  CIRCTSupport
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
