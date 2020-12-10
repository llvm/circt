file(GLOB globbed *.cpp)
add_circt_dialect_library(MLIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTL

  DEPENDS
  MLIRRTLIncGen
  MLIRRTLEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
   )

add_dependencies(circt-headers MLIRRTLIncGen MLIRRTLEnumsIncGen)
