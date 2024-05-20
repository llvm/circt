add_circt_dialect_library(CIRCTVerifTransforms
  VerifyClockedAssertLike.cpp

  DEPENDS
  CIRCTVerifTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTLTL
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)