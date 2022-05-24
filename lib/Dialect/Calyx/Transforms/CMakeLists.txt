add_circt_dialect_library(CIRCTCalyxTransforms
  CompileControl.cpp
  GICM.cpp
  GoInsertion.cpp
  ClkResetInsertion.cpp
  RemoveGroups.cpp
  CalyxHelpers.cpp
  CalyxLoweringUtils.cpp
  MaterializeFSM.cpp

  DEPENDS
  CIRCTCalyxTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTComb
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  )
