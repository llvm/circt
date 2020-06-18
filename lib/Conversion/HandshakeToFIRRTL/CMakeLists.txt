add_mlir_library(MLIRHandshakeToFIRRTL
  HandshakeToFIRRTL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToFIRRTL

  LINK_LIBS
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRHandshakeOps
  MLIRFIRRTL
  )
