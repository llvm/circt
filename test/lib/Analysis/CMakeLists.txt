add_mlir_library(CIRCTAnalysisTest
  TestBitwidthAnalysis.cpp

  LINK_LIBS PUBLIC
  CIRCTAnalysis
  MLIRPass
  )
