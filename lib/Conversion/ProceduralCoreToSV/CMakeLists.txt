add_circt_conversion_library(CIRCTProceduralCoreToSV
  ProceduralCoreToSV.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSim
  CIRCTSV
  MLIRTransforms
)
