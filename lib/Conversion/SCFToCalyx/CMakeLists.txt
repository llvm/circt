add_circt_conversion_library(CIRCTSCFToCalyx
  SCFToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTStaticLogicOps
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRFunc
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  MLIRSCF
)
