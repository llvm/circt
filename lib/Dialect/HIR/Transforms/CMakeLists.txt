add_circt_dialect_library(CIRCTHIRTransforms
  #pass/BitwidthReductionPass.cpp
  pass/MemrefLoweringPass.cpp
  pass/LoopUnrollPass.cpp
  verification/ScheduleVerifier.cpp
  PassRegistration.cpp
  pass/PassPipelines.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
