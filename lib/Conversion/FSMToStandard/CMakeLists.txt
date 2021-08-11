add_mlir_conversion_library(CIRCTFSMToStandard
  FSMToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${PLATO_INCLUDE_DIR}/circt/Conversion/FSMToStandard

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRStandard
  MLIRSupport
  MLIRTransforms
  )
