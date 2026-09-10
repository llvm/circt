add_circt_dialect_library(CIRCTMSFTTransforms
  MSFTExportTcl.cpp
  MSFTLowerInstances.cpp
  MSFTLowerConstructs.cpp
  PassCommon.cpp

  DEPENDS
  CIRCTMSFTTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTEmit
  CIRCTMSFT
  CIRCTHW
  CIRCTSV
  CIRCTSupport
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
