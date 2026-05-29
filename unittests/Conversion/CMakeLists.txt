add_subdirectory(Utils)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_subdirectory(ImportVerilog)
endif()
