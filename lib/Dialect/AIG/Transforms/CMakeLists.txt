add_circt_dialect_library(CIRCTAIGTransforms
  AIGERRunner.cpp
  LowerVariadic.cpp
  LowerWordToBits.cpp

  DEPENDS
  CIRCTAIGPassesIncGen

  LINK_LIBS PUBLIC
  CIRCTAIG
  CIRCTComb
  CIRCTExportAIGER
  CIRCTHW
  CIRCTImportAIGER
  CIRCTSeq
)
