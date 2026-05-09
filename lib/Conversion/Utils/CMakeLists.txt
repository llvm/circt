add_circt_library(CIRCTSVLoweringUtils
  SVLoweringUtils.cpp

  LINK_LIBS PUBLIC
  CIRCTEmit
  CIRCTHW
  CIRCTSV
  MLIRIR
)
