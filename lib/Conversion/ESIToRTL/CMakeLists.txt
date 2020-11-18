add_mlir_conversion_library(CIRCTESIToRTL
  ESIToRTL.cpp

  DEPENDS
  MLIRESIIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRESI
  MLIRRTL
  MLIRTransforms
)
