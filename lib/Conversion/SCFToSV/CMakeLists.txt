add_circt_conversion_library(CIRCTSCFToSV
  SCFToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSV
  MLIRSCFDialect
  MLIRTransforms
)
