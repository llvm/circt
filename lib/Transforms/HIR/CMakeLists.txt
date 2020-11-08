add_mlir_library(MLIRHIRBitwidthReductionPass
  BitwidthReductionPass.cpp

  LINK_LIBS PUBLIC
  MLIRHIR
  )
