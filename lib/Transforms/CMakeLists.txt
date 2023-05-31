add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp
  MapArithToComb.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransforms
  MLIRArithDialect
  CIRCTHW
  CIRCTComb

  DEPENDS
  CIRCTTransformsPassIncGen
)
