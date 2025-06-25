add_circt_dialect_library(CIRCTKanagawaTransforms
  KanagawaCallPrep.cpp
  KanagawaContainerize.cpp
  KanagawaTunneling.cpp
  KanagawaPortrefLowering.cpp
  KanagawaCleanSelfdrivers.cpp
  KanagawaContainersToHW.cpp
  KanagawaArgifyBlocksPass.cpp
  KanagawaReblockPass.cpp
  KanagawaInlineSBlocksPass.cpp
  KanagawaConvertCFToHandshake.cpp
  KanagawaPassPipelines.cpp
  KanagawaPrepareScheduling.cpp
  KanagawaConvertHandshakeToDC.cpp
  KanagawaMethodsToContainers.cpp
  KanagawaAddOperatorLibrary.cpp
  KanagawaEliminateRedundantOps.cpp

  DEPENDS
  CIRCTKanagawaTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDC
  CIRCTHandshake
  CIRCTPipelineOps
  CIRCTCFToHandshake
  CIRCTHandshakeToDC
  CIRCTKanagawa
  CIRCTSSP
  CIRCTHW
  CIRCTHWTransforms
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
