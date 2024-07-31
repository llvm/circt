add_circt_public_c_api_library(CIRCTCAPIExportFIRRTL
  ExportFIRRTL.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTExportFIRRTL
  )
