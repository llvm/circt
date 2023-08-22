add_circt_dialect_library(CIRCTSeqTransforms
  ExternalizeClockGate.cpp
  LowerSeqHLMem.cpp
  LowerSeqFIFO.cpp

  DEPENDS
  CIRCTSeqTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
