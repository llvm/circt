add_circt_conversion_library(CIRCTProbeToSV
  ProbeToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTProbe
  CIRCTSV
)
