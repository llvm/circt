add_subdirectory(circt-bmc)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_subdirectory(circt-verilog-lsp-server)
endif()
