add_circt_library(CIRCTSynthesisTransforms
  SynthesisPipeline.cpp
  TechMapper.cpp

  DEPENDS
  CIRCTSynthesisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTAIGTransforms
  CIRCTCombToAIG
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTSynthesis

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
