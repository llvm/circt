set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(gentool
 gentool.cpp
)
llvm_update_compile_flags(gentool)
target_link_libraries(gentool PRIVATE
  CIRCTExportVerilog
  CIRCTImportFIRRTL
  CIRCTFIRRTLToRTL
  CIRCTFIRRTLTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
  )
