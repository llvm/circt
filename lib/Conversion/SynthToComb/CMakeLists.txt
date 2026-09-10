add_circt_conversion_library(CIRCTSynthToComb
  SynthToComb.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTSynth
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
