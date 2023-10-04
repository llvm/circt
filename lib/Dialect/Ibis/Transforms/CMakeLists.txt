add_circt_dialect_library(CIRCTIbisTransforms
  IbisCallPrep.cpp
  IbisContainerize.cpp
  IbisTunneling.cpp
  IbisPortrefLowering.cpp
  IbisCleanSelfdrivers.cpp
  IbisContainersToHW.cpp
  IbisArgifyBlocksPass.cpp
  IbisReblockPass.cpp
  IbisInlineSBlocksPass.cpp
  IbisConvertCFToHandshake.cpp
  IbisPassPipelines.cpp
  IbisPrepareScheduling.cpp
  IbisConvertHandshakeToDC.cpp
  IbisMethodsToContainers.cpp
  IbisAddOperatorLibrary.cpp

  DEPENDS
  CIRCTIbisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDC
  CIRCTHandshake
  CIRCTPipelineOps
  CIRCTCFToHandshake
  CIRCTHandshakeToDC
  CIRCTIbis
  CIRCTSSP
  CIRCTHW
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
