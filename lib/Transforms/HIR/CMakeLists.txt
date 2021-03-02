add_circt_dialect_library(CIRCTHIRTransforms
  BitwidthReductionPass.cpp
  CanonicalizePass.cpp

  DEPENDS
  CIRCTHIRTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHIR
  )
