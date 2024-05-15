add_circt_public_c_api_library(CIRCTCAPIFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTFirtool
  )
