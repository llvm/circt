add_circt_library(CIRCTStandardToHandshake
  StandardToHandshake.cpp
  InsertMergeBlocks.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToHandshake

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTHandshakeTransforms
  CIRCTSupport
  CIRCTControlFlowLoopAnalysis
  MLIRIR
  MLIRPass
  MLIRArithmeticDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
  )
