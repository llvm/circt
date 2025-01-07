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
  CIRCTSCFToSV
  CIRCTSeqToSV
  CIRCTSimToSV
  CIRCTSeqTransforms
  CIRCTSimTransforms
  CIRCTSVTransforms
  CIRCTTransforms
  CIRCTVerifToSV
  CIRCTVerifTransforms

  MLIRIR
  MLIRSupport
  MLIRTransforms
)
