add_circt_dialect_library(CIRCTCalyxTransforms
  CompileControl.cpp
  GICM.cpp
  GoInsertion.cpp
  ClkResetInsertion.cpp
  RemoveGroups.cpp
  CalyxHelpers.cpp
  CalyxLoweringUtils.cpp

  DEPENDS
  CIRCTCalyxTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTComb
  CIRCTHW
  CIRCTStaticLogicOps
  CIRCTSupport
  MLIRArithmetic
  MLIRFunc
  MLIRIR
  MLIRPass
  MLIRSCF
  MLIRTransformUtils
  )
