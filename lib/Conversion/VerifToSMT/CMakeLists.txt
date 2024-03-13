add_circt_conversion_library(CIRCTVerifToSMT
  VerifToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTHWToSMT
  CIRCTSMT
  CIRCTVerif
  MLIRArithDialect
  MLIRTransforms
  MLIRTransformUtils
  MLIRReconcileUnrealizedCasts
)
