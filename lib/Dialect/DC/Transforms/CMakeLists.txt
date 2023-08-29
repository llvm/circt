add_circt_dialect_library(CIRCTDCTransforms
  DCMaterialization.cpp
  DCTestSCFToDC.cpp

  DEPENDS
  CIRCTDCTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTDC
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
