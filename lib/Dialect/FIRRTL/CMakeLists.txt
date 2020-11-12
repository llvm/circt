file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRFIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/FIRRTL

  DEPENDS
  MLIRFIRRTLIncGen
  MLIRFIRRTLEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRTL
  MLIRSV
   )

add_dependencies(mlir-headers MLIRFIRRTLIncGen MLIRFIRRTLEnumsIncGen)
