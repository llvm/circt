add_circt_dialect_library(CIRCTVerifTransforms
  LowerContracts.cpp
  LowerFormalToHW.cpp
  LowerSymbolicValues.cpp
  PrepareForFormal.cpp
  SimplifyAssumeEq.cpp
  StripContracts.cpp
  VerifyClockedAssertLike.cpp

  DEPENDS
  CIRCTVerifTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTLTL
  CIRCTSupport
  CIRCTSV
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
)
