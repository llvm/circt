add_circt_dialect_library(CIRCTFIRRTLTransforms
  LowerTypes.cpp

  DEPENDS
  CIRCTFIRRTLTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
