add_circt_dialect_library(CIRCTEmitTransforms
  StripEmit.cpp

  DEPENDS
  CIRCTEmitTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTEmit
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
