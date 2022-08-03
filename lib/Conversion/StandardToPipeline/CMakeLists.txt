add_circt_library(CIRCTStandardToPipeline
  StandardToPipeline.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/StandardToPipeline

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRSupport
  MLIRTransforms
  CIRCTPipelineOps
  )
