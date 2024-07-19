add_circt_library(CIRCTCFToHandshake
  CFToHandshake.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/CFToHandshake

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_LIBS PUBLIC
  CIRCTHandshake
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineToStandard
)
