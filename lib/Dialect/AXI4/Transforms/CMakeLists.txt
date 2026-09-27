add_circt_dialect_library(CIRCTAXI4Transforms
  VerifyAXI4Networks.cpp

  DEPENDS
  CIRCTAXI4TransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTAXI4
  CIRCTSupport
  MLIRIR
  MLIRPass
)
