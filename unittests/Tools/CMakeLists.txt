add_subdirectory(circt-bmc)
add_subdirectory(circt-domain-report-server)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_subdirectory(circt-verilog-lsp-server)
endif()
