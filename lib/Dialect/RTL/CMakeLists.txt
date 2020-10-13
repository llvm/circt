file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/RTL

  DEPENDS
  MLIRRTLIncGen
  MLIRRTLEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
   )

add_dependencies(mlir-headers MLIRRTLIncGen MLIRRTLEnumsIncGen)
