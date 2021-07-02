add_circt_library(CIRCTTransforms
  SimpleCanonicalizer.cpp
  CombStrengthReduction.cpp

  DEPENDS
  CIRCTTransformsIncGen
  CIRCTComb

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRRewrite
  MLIRTransformUtils
)

