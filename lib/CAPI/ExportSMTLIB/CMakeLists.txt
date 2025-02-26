add_circt_public_c_api_library(CIRCTCAPIExportSMTLIB
  ExportSMTLIB.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTExportSMTLIB
  )
