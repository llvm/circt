add_circt_dialect_library(CIRCTAIGTransforms
  LowerVariadic.cpp
  LowerWordToBits.cpp
  BalanceVariadic.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTComb
  CIRCTHW
  CIRCTOpDepthAnalysis
)
