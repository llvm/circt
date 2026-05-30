set(LLVM_LINK_COMPONENTS
  Support
)

add_llvm_executable(circt-standalone-opt
  circt-standalone-opt.cpp

  DEPENDS
  SUPPORT_PLUGINS
)

llvm_update_compile_flags(circt-standalone-opt)

target_link_libraries(circt-standalone-opt
  PRIVATE
  CIRCTStandalone
  CIRCTHW
  MLIRIR
  MLIROptLib
  MLIRParser
  MLIRSupport
  MLIRTransforms
)

export_executable_symbols_for_plugins(circt-standalone-opt)
