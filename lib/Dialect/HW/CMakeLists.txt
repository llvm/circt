file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTHW
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/HW

  DEPENDS
  MLIRHWIncGen
  MLIRHWEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
   )

add_dependencies(circt-headers MLIRHWIncGen MLIRHWEnumsIncGen)
