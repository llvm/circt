add_circt_public_c_api_library(CIRCTCAPIRtgTool
  RtgTool.cpp

  LINK_LIBS PUBLIC
  CIRCTRtgToolLib
  MLIRCAPIIR
)
