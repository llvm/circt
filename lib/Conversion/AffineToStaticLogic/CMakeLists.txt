add_circt_library(CIRCTAffineToStaticLogic
  AffineToStaticLogic.cpp

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTStaticLogicOps
  )
