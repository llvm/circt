add_circt_dialect_library(CIRCTLLHD
  LLHDDialect.cpp
  LLHDOps.cpp
  LLHDCanonicalization.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/Dialect/LLHD

  DEPENDS
  CIRCTLLHDEnumsIncGen
  CIRCTLLHDCanonicalizationIncGen
  MLIRLLHDIncGen
  
  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  MLIRStandard
)
