add_circt_dialect_library(CIRCTCalyxTransforms
  GoInsertion.cpp

  DEPENDS
  CIRCTCalyxTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
