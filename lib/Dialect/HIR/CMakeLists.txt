add_subdirectory(Verification)
add_circt_dialect_library(CIRCTHIR
  helper.cpp  
  HIR.cpp  
  HIRDialect.cpp  
  HIROpSyntax.cpp  
  HIROpVerifier.cpp  
  HIRTypeSyntax.cpp
  HIRBuilders.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/HIR

  DEPENDS
  MLIRHIRIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  )
