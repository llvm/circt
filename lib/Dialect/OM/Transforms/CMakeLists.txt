add_circt_dialect_library(CIRCTOMTransforms
  ElaborateObject.cpp
  FreezePaths.cpp
  LinkModules.cpp
  StripOM.cpp

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
