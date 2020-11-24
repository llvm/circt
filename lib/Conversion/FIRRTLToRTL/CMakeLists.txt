add_mlir_conversion_library(MLIRFIRRTLToRTL
  LowerToRTL.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRFIRRTL
  MLIRRTL
  MLIRSV
  MLIRTransforms
)
