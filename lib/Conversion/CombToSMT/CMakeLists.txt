add_circt_conversion_library(CIRCTCombToSMT
  CombToSMT.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHWToSMT
  MLIRSMT
  MLIRTransforms
)
