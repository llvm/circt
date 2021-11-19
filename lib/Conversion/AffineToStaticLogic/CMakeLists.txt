add_circt_library(CIRCTAffineToStaticLogic
  AffineToStaticLogic.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTStaticLogicOps
  )
