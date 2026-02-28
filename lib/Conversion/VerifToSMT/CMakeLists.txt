add_circt_conversion_library(CIRCTVerifToSMT
  VerifToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTHWToSMT
  CIRCTVerif
  MLIRArithDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSMT
  MLIRTransforms
  MLIRTransformUtils
  MLIRReconcileUnrealizedCasts
)
