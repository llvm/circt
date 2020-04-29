set(LIBS
  MLIRTranslation
  MLIRIR
  MLIRSupport

  CIRTEmitVerilog
  CIRTFIRParser
)
add_llvm_tool(cirt-translate
  cirt-translate.cpp
)
llvm_update_compile_flags(cirt-translate)
target_link_libraries(cirt-translate PRIVATE ${LIBS} LLVMSupport)
