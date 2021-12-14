add_circt_library(CIRCTHandshakeToFIRRTL
  HandshakeToFIRRTL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToFIRRTL

  LINK_LIBS PUBLIC
  CIRCTFIRRTL
  CIRCTHandshake
  CIRCTHandshakeTransforms
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
