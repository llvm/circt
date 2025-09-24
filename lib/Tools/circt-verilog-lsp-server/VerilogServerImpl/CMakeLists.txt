# Include the Slang compiler options
include(SlangCompilerOptions)

add_circt_library(CIRCTVerilogLspServerImpl
  VerilogServer.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/circt/Tools/circt-verilog-lsp-server

  DEPENDS
  slang_slang

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLspServerSupportLib
  CIRCTVerilogLspServerUtils
  PRIVATE
  slang_slang
)
