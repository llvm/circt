add_circt_dialect_library(CIRCTVerifTransforms
PrepareForFormal.cpp
  VerifyClockedAssertLike.cpp

  DEPENDS
  CIRCTVerifTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTVerif
  CIRCTLTL
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
