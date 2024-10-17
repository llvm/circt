add_circt_dialect_library(CIRCTAIGTransforms
  GreedyCutDecomp.cpp
  LowerCutToLUT.cpp
  LowerVariadic.cpp
  LowerWordToBits.cpp

  DEPENDS
  CIRCTAIGTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTComb
  CIRCTHW
)
