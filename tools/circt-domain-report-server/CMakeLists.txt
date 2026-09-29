set(LLVM_LINK_COMPONENTS Support)

add_subdirectory(lib)

add_circt_tool(circt-domain-report-server Main.cpp)
target_link_libraries(circt-domain-report-server PRIVATE CIRCTDomainReport)
llvm_update_compile_flags(circt-domain-report-server)
