add_circt_library(CIRCTAffineToStaticLogic
  AffineToStaticLogic.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTStaticLogicOps
  )
