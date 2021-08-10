add_mlir_conversion_library(CIRCTFSMToHW
  FSMToHW.cpp

  ADDITIONAL_HEADER_DIRS
  ${PLATO_INCLUDE_DIR}/circt/Conversion/FSMToHW

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
