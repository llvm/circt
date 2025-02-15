add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp
  MapArithToComb.cpp
  MaximizeSSA.cpp
  InsertMergeBlocks.cpp
  PrintOpCount.cpp
  MemoryBanking.cpp
  IndexSwitchToIf.cpp
  HierarchicalRunner.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTOpCountAnalysis
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRSupport
  MLIRTransforms
  MLIRAffineDialect
  MLIRSCFDialect

  DEPENDS
  CIRCTTransformsPassIncGen
  MLIRArithDialect
)
