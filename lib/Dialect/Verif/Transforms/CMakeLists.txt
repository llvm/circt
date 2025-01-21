add_circt_dialect_library(CIRCTVerifTransforms
  VerifyClockedAssertLike.cpp
  PrepareForFormal.cpp
  LowerFormalToHW.cpp
  LowerContracts.cpp
  SimplifyAssumeEq.cpp

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
