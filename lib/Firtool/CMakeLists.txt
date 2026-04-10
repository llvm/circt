add_circt_library(CIRCTFirtool
  Firtool.cpp

  LINK_LIBS PUBLIC
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTHWToBTOR2
  CIRCTHWToSV
  CIRCTHWTransforms
  CIRCTLTLToCore
  CIRCTOMTransforms
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
