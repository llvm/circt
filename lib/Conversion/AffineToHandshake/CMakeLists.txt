add_mlir_library(MLIRAffineToHandshake
  AffineToHandshake.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToHandshake

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandardOps
  MLIRAffineOps
  MLIRAffineUtils
  MLIRSupport
  MLIRTransforms
  MLIRHandshakeOps
  )
