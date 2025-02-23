add_circt_conversion_library(CIRCTICE40ToBLIF
  ICE40ToBLIF.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTBLIF
  CIRCTHW
  CIRCTICE40
  CIRCTSupport
  MLIRTransforms
)
