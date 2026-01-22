add_circt_dialect_library(CIRCTLTLTransforms
  InferLTLClocks.cpp

  DEPENDS
  CIRCTLTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTLTL
  CIRCTHW
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
