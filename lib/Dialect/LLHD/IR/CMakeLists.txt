add_circt_dialect_library(MLIRLLHD
  LLHDDialect.cpp
  LLHDOps.cpp
  LLHDCanonicalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/LLHD

  DEPENDS
  MLIRLLHDEnumsIncGen
  MLIRLLHDIncGen
  MLIRLLHDCanonicalizationIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  MLIRStandard
)
