add_circt_dialect_library(CIRCTAIGTransforms
  LowerVariadic.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTComb
  CIRCTHW
)
