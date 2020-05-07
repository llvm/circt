file(GLOB globbed *.cpp)
add_mlir_dialect_library(MLIRRTL
  ${globbed}

  ADDITIONAL_HEADER_DIRS
  ${CIRT_MAIN_INCLUDE_DIR}/cirt/Dialect/RTL

  DEPENDS
  MLIRRTLIncGen

  LINK_LIBS PUBLIC
  LLVMSupport
   )

add_dependencies(mlir-headers MLIRRTLIncGen)
