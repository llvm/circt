add_circt_library(CIRCTStandardToHandshake
  StandardToHandshake.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToHandshake

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRArithmetic
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
