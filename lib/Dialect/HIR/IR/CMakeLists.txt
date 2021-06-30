add_circt_dialect_library(CIRCTHIR
  helper.cpp  
  HIROpSyntax.cpp  
  HIRDialect.cpp  
  HIRInterfaces.cpp
  HIR.cpp  
  HIROpVerifier.cpp  
  HIROpCanonicalizer.cpp
  HIRTypeSyntax.cpp
  HIRBuilders.cpp
  HIROpExtraClassDeclaration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/lib/Dialect/HIR

  DEPENDS
  MLIRHIRIncGen
  MLIRHIRInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  MLIRStandard
  )
