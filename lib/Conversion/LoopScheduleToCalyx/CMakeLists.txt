add_circt_conversion_library(CIRCTLoopScheduleToCalyx
  LoopScheduleToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithOpsInterfacesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTCalyxTransforms
  CIRCTLoopSchedule
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
)
