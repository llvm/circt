add_circt_dialect_library(CIRCTICE40
  ICE40Ops.cpp
  ICE40Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/ICE40

  DEPENDS
  MLIRICE40IncGen
  
  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
   )

add_dependencies(circt-headers MLIRICE40IncGen)
