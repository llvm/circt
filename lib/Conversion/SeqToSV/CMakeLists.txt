add_circt_conversion_library(CIRCTSeqToSV
  FirMemLowering.cpp
  FirRegLowering.cpp
  SeqToSV.cpp
  LowerFirMem.cpp

  DEPENDS
  CIRCTConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  CIRCTSeq
)
