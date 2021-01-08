file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTSV
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/SV

  DEPENDS
  MLIRSVIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTRTL
   )

add_dependencies(circt-headers MLIRSVIncGen)
