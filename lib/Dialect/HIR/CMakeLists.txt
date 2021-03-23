add_subdirectory(Verification)
#add_compile_options(-Wall -Wextra)
file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTHIR
  ${globbed}

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
