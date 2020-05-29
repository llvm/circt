file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRFIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRT_MAIN_INCLUDE_DIR}/cirt/Dialect/FIRRTL

  DEPENDS
  MLIRFIRRTLIncGen
  MLIRFIRRTLEnumsIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
   )

add_dependencies(mlir-headers MLIRFIRRTLIncGen MLIRFIRRTLEnumsIncGen)
