add_circt_dialect_library(CIRCTFIRRTLTransforms
  BlackBoxMemory.cpp
  BlackBoxReader.cpp
  CheckCombCycles.cpp
  CreateSiFiveMetadata.cpp
  EmitOMIR.cpp
  ExpandWhens.cpp
  GrandCentral.cpp
  GrandCentralTaps.cpp
  GrandCentralSignalMappings.cpp
  IMConstProp.cpp
  InferResets.cpp
  InferReadWrite.cpp
  InferWidths.cpp
  LowerAnnotations.cpp
  LowerCHIRRTL.cpp
  LowerTypes.cpp
  ModuleInliner.cpp
  PrefixModules.cpp
  PrintInstanceGraph.cpp
  RemoveResets.cpp
  RemoveUnusedPorts.cpp
  WireDFT.cpp

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
