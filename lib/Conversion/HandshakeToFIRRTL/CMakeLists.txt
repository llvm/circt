add_circt_library(MLIRHandshakeToFIRRTL
  HandshakeToFIRRTL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToFIRRTL

  DEPENDS
  MLIRHandshakeRewritersIncGen
  MLIRHandshakeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRHandshakeOps
  MLIRFIRRTL
  )
