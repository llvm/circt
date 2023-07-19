add_circt_library(CIRCTAffineToLoopSchedule
  AffineToLoopSchedule.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithOpsInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRAffineToStandard
  MLIRPass
  CIRCTScheduling
  CIRCTSchedulingAnalysis
  CIRCTLoopSchedule
  )
