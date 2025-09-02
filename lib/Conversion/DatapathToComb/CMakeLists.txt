add_circt_conversion_library(CIRCTDatapathToComb
  DatapathToComb.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTAIGAnalysis
  CIRCTComb
  CIRCTDatapath
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
