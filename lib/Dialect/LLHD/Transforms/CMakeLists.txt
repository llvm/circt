add_circt_dialect_library(MLIRLLHDTransforms
  TemporalRegions.cpp
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp
  MemoryToBlockArgumentPass.cpp
  EarlyCodeMotionPass.cpp

  DEPENDS
  MLIRLLHDTransformsIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRStandard
  MLIRLLHD
  MLIRTransformUtils
)
