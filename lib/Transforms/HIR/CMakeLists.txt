add_mlir_library(CIRCTHIRBitwidthReductionPass
  BitwidthReductionPass.cpp

  LINK_LIBS PUBLIC
  CIRCTHIR
  )
