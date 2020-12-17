add_circt_library(MLIRStandardToStaticLogic
  StandardToStaticLogic.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToStaticLogic

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  MLIRStaticLogicOps
  )
