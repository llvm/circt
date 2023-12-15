add_mlir_public_c_api_library(CIRCTCAPIConversion
  Passes.cpp

  LINK_LIBS PUBLIC
  CIRCTAffineToLoopSchedule
  CIRCTArcToLLVM
  CIRCTCalyxToFSM
  CIRCTCalyxToHW
  CIRCTCalyxNative
  CIRCTCombToArith
  CIRCTCombToLLVM
  CIRCTConvertToArcs
  CIRCTDCToHW
  CIRCTExportChiselInterface
  CIRCTExportVerilog
  CIRCTFIRRTLToHW
  CIRCTFSMToSV
  CIRCTHandshakeToDC
  CIRCTHandshakeToHW
  CIRCTHWArithToHW
  CIRCTHWToLLHD
  CIRCTHWToLLVM
  CIRCTHWToBTOR2
  CIRCTHWToSV
  CIRCTHWToSystemC
  CIRCTLLHDToLLVM
  CIRCTLoopScheduleToCalyx
  CIRCTMooreToCore
  CIRCTPipelineToHW
  CIRCTSCFToCalyx
  CIRCTSeqToSV
  CIRCTCFToHandshake
  CIRCTVerifToSV
)

