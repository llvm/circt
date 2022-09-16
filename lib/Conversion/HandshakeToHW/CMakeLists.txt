add_circt_library(CIRCTHandshakeToHW
  HandshakeToHW.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/HandshakeToHW

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTESI
  CIRCTHandshake
  CIRCTHandshakeTransforms
  MLIRIR
  MLIRPass
  MLIRArithmeticDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  )
