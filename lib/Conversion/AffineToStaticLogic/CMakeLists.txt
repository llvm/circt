add_circt_library(CIRCTAffineToStaticLogic
  AffineToStaticLogic.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStaticLogic

  LINK_LIBS PUBLIC
  MLIRPass
  CIRCTScheduling
  CIRCTDependenceAnalysis
  )
