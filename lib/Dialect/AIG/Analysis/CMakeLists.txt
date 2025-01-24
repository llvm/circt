add_circt_dialect_library(CIRCTAIGAnalysis
  ResourceUsageAnalysis.cpp
  LongestPathAnalysis.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  MLIRSupport
  CIRCTAIG
  CIRCTComb
  CIRCTHW
)
