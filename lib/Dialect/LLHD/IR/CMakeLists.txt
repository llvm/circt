add_mlir_dialect_library(MLIRLLHD
  LLHDDialect.cpp
  LLHDOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/LLHD

  DEPENDS
  MLIRLLHDEnumsIncGen
  MLIRLLHDIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
)
