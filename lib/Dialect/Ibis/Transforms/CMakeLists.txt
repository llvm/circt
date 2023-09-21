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

  DEPENDS
  CIRCTIbisTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTDC
  CIRCTIbis
  CIRCTHW
  CIRCTSupport
  CIRCTTransforms
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
