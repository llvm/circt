add_circt_dialect_library(CIRCTHIRTransforms
  #pass/BitwidthReductionPass.cpp
  Passes/IndexLoweringPass.cpp
  Passes/LoopUnrollPass.cpp
  Passes/MemrefLoweringPass.cpp
  Passes/MemrefLoweringUtils.cpp
  Passes/PassPipelines.cpp
  Passes/RegisterAllocationPass.cpp
  Passes/ScheduleVerifier.cpp
  Passes/SeqScheduler.cpp
  Passes/SimplifyCtrl.cpp
  Passes/SimplifyCtrlUtils.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIR
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  CIRCTHIRAnalysis
  )
