add_circt_conversion_library(CIRCTDatapathToComb
  DatapathToComb.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTDatapath
  CIRCTComb
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
