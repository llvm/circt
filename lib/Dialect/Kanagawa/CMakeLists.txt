add_circt_dialect_library(CIRCTKanagawa
  KanagawaDialect.cpp
  KanagawaOps.cpp
  KanagawaTypes.cpp

  DEPENDS
  MLIRKanagawaIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  CIRCTHW
  CIRCTDC
  CIRCTSeq
)

add_subdirectory(Transforms)
