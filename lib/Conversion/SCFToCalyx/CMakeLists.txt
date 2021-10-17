add_circt_conversion_library(CIRCTSCFToCalyx
  SCFToCalyx.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  MLIRSCF
)
