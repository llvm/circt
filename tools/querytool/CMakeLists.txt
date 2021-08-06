set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(querytool
 querytool.cpp
 Lexer.cpp
 Parser.cpp
)
llvm_update_compile_flags(querytool)
target_link_libraries(querytool PRIVATE
  CIRCTQuery

  CIRCTExportVerilog
  CIRCTImportFIRFile
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTSVTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
  )
