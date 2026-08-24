add_circt_dialect_library(CIRCTPIR
  PIRDialect.cpp
  PIROps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/PIR

  DEPENDS
  MLIRPIRIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  MLIRInferTypeOpInterface
  MLIRIR
)
