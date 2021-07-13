set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_tool(querytool
 querytool.cpp
)
llvm_update_compile_flags(querytool)
target_link_libraries(querytool PRIVATE
  CIRCTImportFIRRTL
  CIRCTFIRRTLToHW
  CIRCTFIRRTLTransforms
  CIRCTQuery
  CIRCTSVTransforms
  CIRCTTransforms

  MLIRParser
  MLIRSupport
  MLIRIR
  MLIROptLib
  MLIRStandard
  MLIRTransforms
  MLIRTranslation
  )
