add_circt_library(CIRCTStandardToHandshake
  StandardToHandshake.cpp
  InsertMergeBlocks.cpp
  MaximizeSSA.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToHandshake

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
