add_circt_dialect_library(CIRCTIbisAnalysis
  Hierarchy.cpp

  LINK_LIBS PUBLIC
  CIRCTIbis
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
