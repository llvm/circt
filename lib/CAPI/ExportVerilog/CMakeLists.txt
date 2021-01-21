add_circt_library(CIRCTCAPIExportVerilog

  ExportVerilog.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir-c

  LINK_LIBS PUBLIC
  MLIRCAPIIR
  CIRCTExportVerilog
  )
