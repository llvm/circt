add_circt_dialect_library(CIRCTRTLTransforms
  GreyBoxer.cpp

  DEPENDS
  CIRCTRTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTRTL
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
