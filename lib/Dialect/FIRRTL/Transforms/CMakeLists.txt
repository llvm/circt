add_circt_dialect_library(CIRCTFIRRTLTransforms
  BlackBoxMemory.cpp
  BlackBoxReader.cpp
  CheckCombCycles.cpp
  ExpandWhens.cpp
  GrandCentral.cpp
  GrandCentralTaps.cpp
  IMConstProp.cpp
  InferResets.cpp
  InferWidths.cpp
  LowerCHIRRTL.cpp
  LowerTypes.cpp
  ModuleInliner.cpp
  PrintInstanceGraph.cpp

  DEPENDS
  CIRCTFIRRTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTHW
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
