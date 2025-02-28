add_subdirectory(Utils)
add_subdirectory(VerilogServerImpl)
add_circt_library(CIRCTVerilogLspServerLib
  CirctVerilogLspServerMain.cpp
  LSPServer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/circt-verilog-lsp-server

  LINK_LIBS PUBLIC
  CIRCTVerilogLspServerImpl
  MLIRLspServerSupportLib
)
