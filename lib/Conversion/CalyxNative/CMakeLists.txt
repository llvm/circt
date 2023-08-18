add_circt_library(CIRCTCalyxNative
  CalyxNative.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/CalyxNative

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTExportCalyx
  MLIRIR
  MLIRPass
  MLIRSupport
  )
