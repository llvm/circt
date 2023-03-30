add_circt_library(CIRCTHandshakeToHW
  DCToHW.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToHW

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTDC
  CIRCTESI
  CIRCTHandshake
  CIRCTHandshakeTransforms
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  )
