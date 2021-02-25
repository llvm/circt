add_subdirectory(Verification)
add_compile_options(-Werror -Wall -Wextra)
add_circt_dialect_library(CIRCTHIR
  HIRDialect.cpp
  HIR.cpp

  DEPENDS
  MLIRHIRIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  )
