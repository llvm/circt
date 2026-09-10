add_circt_public_c_api_library(CIRCTCAPITransforms
  Transforms.cpp

  LINK_LIBS PUBLIC
  CIRCTTransforms
)
