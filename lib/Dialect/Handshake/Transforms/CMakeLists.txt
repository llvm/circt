add_circt_dialect_library(CIRCTHandshakeTransforms
  HandshakeWrapESIPass.cpp

  DEPENDS
  CIRCTHandshakeTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  CIRCTESI
  CIRCTHW
)
