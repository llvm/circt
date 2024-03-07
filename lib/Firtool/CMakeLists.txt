add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTLTLToCore
  CIRCTHWToBTOR2
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSimToSV
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTVerifToSV

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
