add_circt_dialect_library(CIRCTAIGAnalysis
  LongestPathAnalysis.cpp
  PrintLongestPathAnalysis.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  CIRCTAIG
  CIRCTComb
  CIRCTCombToAIG
  CIRCTTransforms
  CIRCTHW
  CIRCTSeq
)
