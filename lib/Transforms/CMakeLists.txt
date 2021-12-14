add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRef
  MLIRStandard
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  CIRCTTransformsPassIncGen
)
