add_circt_dialect_library(CIRCTHIRTransforms
  #pass/BitwidthReductionPass.cpp
  Passes/IndexLoweringPass.cpp
  Passes/RegisterAllocationPass.cpp
  Passes/SeqScheduler.cpp
  Passes/MemrefLoweringPass.cpp
  Passes/LoopUnrollPass.cpp
  Passes/ScheduleVerifier.cpp
  Passes/PassPipelines.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen
  CIRCTHIRAnalysis

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
