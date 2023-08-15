add_circt_dialect_library(CIRCTIbisTransforms
  IbisCallPrep.cpp
  IbisContainerize.cpp

  DEPENDS
  CIRCTIbisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDC
  CIRCTIbis
  CIRCTHW
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
