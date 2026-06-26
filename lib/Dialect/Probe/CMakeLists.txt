add_circt_dialect_library(CIRCTProbe
  ProbeDialect.cpp
  ProbeOps.cpp
  ProbeTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${CIRCT_MAIN_INCLUDE_DIR}/circt/Dialect/Probe

  DEPENDS
  MLIRProbeIncGen

  LINK_COMPONENTS
  Support

  LINK_LIBS PUBLIC
  CIRCTHW
  CIRCTSeq
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
)
