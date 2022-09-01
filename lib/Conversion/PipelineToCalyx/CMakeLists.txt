add_circt_conversion_library(CIRCTPipelineToCalyx
  PipelineToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTCalyxTransforms
  CIRCTPipelineOps
  MLIRIR
  MLIRPass
  MLIRArithmeticDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
)
