add_circt_library(CIRCTSynthesis
  SynthesisPipeline.cpp

  LINK_LIBS PUBLIC
  CIRCTAIGTransforms
  CIRCTCombToAIG
  CIRCTHWTransforms
  CIRCTTransforms
  CIRCTSupport

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
