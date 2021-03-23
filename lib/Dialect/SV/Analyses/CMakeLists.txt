add_circt_dialect_library(CIRCTSVAnalyses
  LegalNames.cpp

  LINK_LIBS PUBLIC
  CIRCTSV
  MLIRIR
)
