add_circt_library(CIRCTTransforms
  Feedthrough.cpp
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTControlFlowLoopAnalysis
  CIRCTHW
  MLIRIR
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
  MLIRTransformUtils

  DEPENDS
  CIRCTTransformsPassIncGen
)
