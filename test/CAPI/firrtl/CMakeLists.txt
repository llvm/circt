add_llvm_executable(circt-capi-firrtl-test
  firrtl.c
)
llvm_update_compile_flags(circt-capi-firrtl-test)

target_link_libraries(circt-capi-firrtl-test
  PRIVATE

  CIRCTCAPIFIRRTL
)
