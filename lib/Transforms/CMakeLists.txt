add_circt_library(CIRCTTransforms
  SimpleCanonicalizer.cpp

  DEPENDS
  CIRCTTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

