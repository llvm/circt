add_circt_dialect_library(CIRCTHIRAnalysis

  FanoutAnalysis.cpp


  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
