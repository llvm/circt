add_subdirectory(arcilator)
add_subdirectory(circt-bmc)
add_subdirectory(circt-lec)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_subdirectory(circt-verilog-lsp-server)
endif()
