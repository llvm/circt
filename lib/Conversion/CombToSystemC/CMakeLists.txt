add_circt_conversion_library(CIRCTCombToSystemC
  CombToSystemC.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSystemC
  CIRCTHW
  CIRCTComb
  MLIRTransforms
)
