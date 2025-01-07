add_circt_conversion_library(CIRCTSimToSV
  PrintsToSV.cpp
  SimToSV.cpp
  TriggersToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTEmit
  CIRCTSim
  CIRCTSeq
)
