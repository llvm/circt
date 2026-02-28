add_circt_conversion_library(CIRCTDatapathToSMT
  DatapathToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTDatapath
  CIRCTComb
  CIRCTHWToSMT
  MLIRSMT
  MLIRTransforms
)
