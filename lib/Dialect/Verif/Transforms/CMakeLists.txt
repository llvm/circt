add_circt_dialect_library(CIRCTVerifTransforms
  LowerContracts.cpp
  LowerFormalToHW.cpp
  PrepareForFormal.cpp
  SimplifyAssumeEq.cpp
  StripContracts.cpp
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
