add_circt_dialect_library(CIRCTAIGTransforms
  GreedyCut.cpp
  LowerCutToLUT.cpp

  DEPENDS
  CIRCTAIGTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
)
