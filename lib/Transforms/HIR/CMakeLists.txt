add_circt_dialect_library(CIRCTHIRTransforms
  #pass/BitwidthReductionPass.cpp
  pass/CanonicalizePass.cpp
  pass/LoopUnrollPass.cpp
  verification/ScheduleVerifier.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformUtils
  CIRCTHIR
  )
