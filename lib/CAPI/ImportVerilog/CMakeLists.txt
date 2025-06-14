add_circt_public_c_api_library(CIRCTCAPIImportVerilog
  ImportVerilog.cpp

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTImportVerilog
  )
