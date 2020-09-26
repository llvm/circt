add_mlir_library(MLIRStandardToHandshake
  StandardToHandshake.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToHandshake

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRHandshakeOps
  MLIRAffineToStandard
  )
