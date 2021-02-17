file(GLOB globbed *.cpp)
add_circt_dialect_library(CIRCTRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTL

  DEPENDS
  MLIRRTLIncGen
  MLIRRTLEnumsIncGen
  CIRCTComb
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTComb
  MLIRIR
   )

add_dependencies(circt-headers MLIRRTLIncGen MLIRRTLEnumsIncGen)
