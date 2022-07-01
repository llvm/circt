add_circt_conversion_library(CIRCTSCFToCalyx
  SCFToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTCalyxTransforms
  CIRCTStaticLogicOps
  MLIRIR
  MLIRPass
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  MLIRSCFDialect
)
