add_circt_dialect_library(CIRCTSeqTransforms
  ExternalizeClockGate.cpp
  HWMemSimImpl.cpp
  LowerSeqHLMem.cpp
  LowerSeqFIFO.cpp
  LowerSeqShiftReg.cpp

  DEPENDS
  CIRCTSeqTransformsIncGen

  LINK_LIBS PUBLIC
  CIRCTComb
  CIRCTEmit
  CIRCTHW
  CIRCTSeq
  CIRCTSupport
  CIRCTSV
  CIRCTVerif
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
