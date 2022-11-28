add_circt_dialect_library(CIRCTSSPTransforms
  PrintInstance.cpp

  DEPENDS
  CIRCTSSPTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTScheduling
  CIRCTSSP
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
