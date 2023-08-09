add_circt_conversion_library(CIRCTIbisToHW
  IbisToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTIbis
  CIRCTIbisAnalysis
  CIRCTSupport
  MLIRTransforms
)
