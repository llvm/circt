add_circt_dialect_library(CIRCTCalyxTransforms
  CalyxPasses.cpp

  DEPENDS
  CIRCTCalyxTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
