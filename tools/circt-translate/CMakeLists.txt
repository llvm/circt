get_property(dialect_libs GLOBAL PROPERTY CIRCT_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY CIRCT_TRANSLATION_LIBS)

if(CIRCT_SLANG_FRONTEND_ENABLED)
  add_compile_definitions(CIRCT_SLANG_FRONTEND_ENABLED)
endif()

if(CIRCT_YOSYS_INTEGRATION_ENABLED)
  add_compile_definitions(CIRCT_YOSYS_INTEGRATION_ENABLED)
endif()

set(LLVM_LINK_COMPONENTS
  Support
)

add_circt_tool(circt-translate
  circt-translate.cpp
)

llvm_update_compile_flags(circt-translate)

target_link_libraries(circt-translate
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  MLIRIR
  MLIRSupport
  MLIRTranslateLib
)
