add_circt_library(CIRCTHandshakeToFIRRTL
  HandshakeToFIRRTL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToFIRRTL

  DEPENDS
  MLIRHandshakeRewritersIncGen
  MLIRHandshakeInterfacesIncGen

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTHandshakeOps
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
