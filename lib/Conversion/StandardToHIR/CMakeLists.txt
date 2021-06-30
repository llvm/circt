add_circt_conversion_library(CIRCTStandardToHIR
  StandardToHIR.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRMemRef
  MLIRStandard
  MLIRTransforms
)
