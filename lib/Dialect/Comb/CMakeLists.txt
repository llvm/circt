file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTComb
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Comb

  DEPENDS
  CIRCTHW
  MLIRCombIncGen
  MLIRCombEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
   )

add_dependencies(circt-headers MLIRCombIncGen MLIRCombEnumsIncGen)
