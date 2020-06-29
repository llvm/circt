add_mlir_dialect_library(MLIRLLHDTransforms
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
