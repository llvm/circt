add_mlir_dialect_library(MLIRLLHDTransforms
  PassRegistration.cpp
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp

  DEPENDS
  MLIRLLHDIncGen
  MLIRLLHDTransformsIncGen

  LINK_LIBS
  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRTransformUtils
  LLVMSupport
)
