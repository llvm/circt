add_circt_public_c_api_library(CIRCTCAPIRTGPipeline
  RTGPipeline.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTRTGDialect
  CIRCTRTGTransforms
  CIRCTEmitRTGAssembly
)
