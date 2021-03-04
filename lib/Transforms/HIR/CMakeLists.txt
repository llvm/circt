add_circt_dialect_library(CIRCTHIRTransforms
  BitwidthReductionPass.cpp
  CanonicalizePass.cpp
  PassRegistration.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHIR
  )
