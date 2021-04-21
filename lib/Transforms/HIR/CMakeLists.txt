add_circt_dialect_library(CIRCTHIRTransforms
  #pass/BitwidthReductionPass.cpp
  pass/CanonicalizePass.cpp
  verification/ScheduleVerifier.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHIR
  )
