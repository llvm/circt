add_circt_dialect_library(CIRCTBLIF
  BLIFOps.cpp
  BLIFDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/BLIF

  DEPENDS
  MLIRBLIFIncGen
  CIRCTBLIFEnumsIncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRIR
  MLIRInferTypeOpInterface
   )

add_dependencies(circt-headers 
  MLIRBLIFIncGen
  CIRCTBLIFEnumsIncGen
  )

  add_subdirectory(Export)
  add_subdirectory(Import)
