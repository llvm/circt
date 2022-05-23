add_circt_library(CIRCTCalyxToFSM
  CalyxToFSM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/CalyxToFSM

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTCalyx
  CIRCTFSM
  MLIRIR
  MLIRPass
  MLIRFunc
  MLIRSupport
  MLIRTransforms
  )
