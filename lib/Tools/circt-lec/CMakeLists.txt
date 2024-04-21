add_circt_library(CIRCTLECTransforms
  ConstructLEC.cpp

  DEPENDS
  CIRCTLECTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTVerif

  MLIRIR
  MLIRSupport
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTransforms
  MLIRTransformUtils
)
