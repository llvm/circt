add_subdirectory(circt-bmc)
add_subdirectory(circt-lec)
add_subdirectory(rtgtool)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_subdirectory(circt-verilog-lsp-server)
endif()
