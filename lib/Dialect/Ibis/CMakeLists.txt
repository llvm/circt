add_circt_dialect_library(CIRCTIbis
  IbisDialect.cpp
  IbisOps.cpp
  IbisTypes.cpp
  NestedSymbolTable.cpp

  DEPENDS
  MLIRIbisIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  CIRCTDC
  CIRCTSeq
)

add_subdirectory(Transforms)
