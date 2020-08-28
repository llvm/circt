file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRSV
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SV

  DEPENDS
  MLIRSVIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
   )

add_dependencies(mlir-headers MLIRSVIncGen)
