llvm_add_library(CIRCTVerilogLspServerLib
  LSPServer.cpp
  VerilogServer.cpp
  Protocol.cpp
  CirctVerilogLspServerMain.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/crct-verilog-lsp-server

  LINK_LIBS PUBLIC
  MLIRLspServerSupportLib
  )
