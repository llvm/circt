add_circt_conversion_library(CIRCTCombToSynth
  CombToSynth.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Conversion/CombToSynth

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTComb
  CIRCTDatapath
  CIRCTSynth
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
)
