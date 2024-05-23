add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWToBTOR2
  CIRCTHWTransforms
  CIRCTLTLToCore
  CIRCTOMTransforms
  CIRCTSeqToSV
  CIRCTSimToSV
  CIRCTSeqTransforms
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTVerifToSV
  CIRCTVerifTransforms

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
