add_circt_conversion_library(CIRCTSimToSV
  SimToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSim
  CIRCTSeq
)
