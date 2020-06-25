set(LIBS

  MLIREDSC
  MLIRIR
  MLIRStandardOps
  MLIRLLHD
  MLIRTransformUtils
  LLVMSupport)

add_llvm_library(MLIRLLHDTransforms
  ProcessLoweringPass.cpp
  FunctionEliminationPass.cpp

  DEPENDS
  MLIRLLHDOpsIncGen
  MLIRLLHDTransformsIncGen
)
target_link_libraries(MLIRLLHDTransforms ${LIBS})
