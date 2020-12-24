add_circt_dialect_library(CIRCTLLHDTransforms
  TemporalRegions.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  EarlyCodeMotionPass.cpp

  DEPENDS
  CIRCTLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandard
  CIRCTLLHD
  MLIRTransformUtils
)
