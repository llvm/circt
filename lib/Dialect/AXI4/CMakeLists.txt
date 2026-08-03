add_circt_dialect_library(CIRCTAXI4
  AXI4Dialect.cpp
  AXI4Ops.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/AXI4

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  MLIRIR
)
