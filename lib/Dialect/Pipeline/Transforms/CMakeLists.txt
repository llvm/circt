add_circt_dialect_library(CIRCTPipelineTransforms
  ExplicitRegs.cpp

  DEPENDS
  CIRCTPipelineTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTPipelineOps
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
