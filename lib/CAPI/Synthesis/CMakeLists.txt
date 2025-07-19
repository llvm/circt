add_circt_public_c_api_library(CIRCTCAPISynthesis
  SynthesisPipeline.cpp

  LINK_LIBS PUBLIC
  CIRCTSynthesisTransforms
  MLIRCAPIIR
)
