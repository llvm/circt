add_circt_conversion_library(CIRCTCombToArith
  CombToArith.cpp

  DEPENDS
  CIRCTConversionPassIncGen
  MLIRArithOpsInterfacesIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  MLIRArithDialect
  MLIRTransforms
)
