add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWTransforms
  CIRCTLTLToCore
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSimToSV
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTVerifToSV
  CIRCTHWToBTOR2

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
