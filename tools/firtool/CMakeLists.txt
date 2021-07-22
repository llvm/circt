set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(firtool
 firtool.cpp
)
llvm_update_compile_flags(firtool)
target_link_libraries(firtool PRIVATE
  CIRCTExportVerilog
  CIRCTImportFIRRTL
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
