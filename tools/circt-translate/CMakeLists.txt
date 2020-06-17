set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(circt-translate
  circt-translate.cpp
)
llvm_update_compile_flags(circt-translate)
target_link_libraries(circt-translate
  PRIVATE
  MLIRTranslation
  MLIRIR
  MLIRStandardOps
  MLIRSupport
 
  CIRCTEmitVerilog
  CIRCTFIRParser
  )
