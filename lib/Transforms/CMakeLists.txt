add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  CIRCTTransformsPassIncGen
)
