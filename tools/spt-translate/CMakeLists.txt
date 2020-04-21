set(LIBS
  MLIRTranslation
  MLIRIR
  MLIRSupport

  SPTEmitVerilog
  SPTFIRParser
)
add_llvm_tool(spt-translate
  spt-translate.cpp
)
llvm_update_compile_flags(spt-translate)
target_link_libraries(spt-translate PRIVATE ${LIBS} LLVMSupport)
