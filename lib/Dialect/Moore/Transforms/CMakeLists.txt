add_circt_dialect_library(CIRCTMooreTransforms
LowerConcatRef.cpp
MergeExtractRef.cpp
SimplifyProcedures.cpp


  DEPENDS
  CIRCTMooreTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTMoore
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
