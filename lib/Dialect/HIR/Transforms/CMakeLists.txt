add_circt_dialect_library(CIRCTHIRTransforms
  Passes/LoopUnrollPass.cpp
  Passes/MemrefLoweringPass.cpp
  Passes/MemrefLoweringUtils.cpp
  Passes/PassPipelines.cpp
  Passes/SimplifyCtrl.cpp
  Passes/SimplifyCtrlUtils.cpp
  Passes/VerifySchedulePass.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIR
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  CIRCTHIRAnalysis
  )
