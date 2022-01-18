add_circt_dialect_library(CIRCTGAATransforms
  GenerateConflictMatrix.cpp
  ReferRules.cpp

  DEPENDS
  CIRCTGAAAttrsIncGen
  CIRCTGAATransformsIncGen
  CIRCTGAAOpInterfacesIncGen
  CIRCTGAA

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  CIRCTGAA
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
