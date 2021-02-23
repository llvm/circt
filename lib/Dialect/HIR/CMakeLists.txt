add_subdirectory(Verification)
add_compile_options(-Werror -Wall -Wextra)
add_mlir_dialect_library(CIRCTHIR
  HIRDialect.cpp
  HIR.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/HIR

  DEPENDS
  MLIRHIRIncGen

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRControlFlowInterfaces
  MLIRCallInterfaces
  )
