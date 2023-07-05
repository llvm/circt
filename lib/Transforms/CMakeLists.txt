set(LLVM_OPTIONAL_SOURCES
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp
  MapArithToComb.cpp
  MaximizeSSA.cpp
  InsertMergeBlocks.cpp
  PrintOpCount.cpp
  MemoryBanking.cpp
  IndexSwitchToIf.cpp
  EggSynthesis.cpp
)

add_circt_library(CIRCTTransforms
  FlattenMemRefs.cpp
  StripDebugInfoWithPred.cpp
  MapArithToComb.cpp

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

if(egg-netlist-synthesizer_FOUND)
  add_circt_library(CIRCTEggSynthesis
    EggSynthesis.cpp

    ADDITIONAL_HEADER_DIRS
    ${CIRCT_MAIN_INCLUDE_DIR}/circt/Transforms

    LINK_LIBS PUBLIC
    CIRCTArc
    CIRCTComb
    MLIRIR
    MLIRSupport
    MLIRTransforms
    egg_netlist_synthesizer

    DEPENDS
    CIRCTTransformsPassIncGen
  )

  target_include_directories(CIRCTEggSynthesis
    PUBLIC
    ${egg-netlist-synthesizer_INCLUDE_DIRS}
  )
endif()
