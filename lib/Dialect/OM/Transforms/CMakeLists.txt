add_circt_dialect_library(CIRCTOMTransforms
  FreezePaths.cpp
  LinkModules.cpp
  StripOM.cpp
  VerifyObjectFields.cpp

  DEPENDS
  CIRCTOMTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTOM
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
