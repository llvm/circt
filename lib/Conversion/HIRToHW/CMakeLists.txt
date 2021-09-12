add_circt_conversion_library(CIRCTHIRToHW
  HIRToHWUtils.cpp
  HIRToHW.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRTransforms
)
