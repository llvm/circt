add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportChiselInterface
  CIRCTSVTransforms
  CIRCTHWTransforms
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTTransforms
  CIRCTSeqTransforms

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
