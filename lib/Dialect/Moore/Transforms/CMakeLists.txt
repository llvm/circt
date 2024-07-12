add_circt_dialect_library(CIRCTMooreTransforms
LowerConcatRef.cpp
PullNonBlockingUp.cpp
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
