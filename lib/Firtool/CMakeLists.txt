add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTVerifToSV

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
